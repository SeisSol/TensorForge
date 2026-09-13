// === base name ===
kernel_6bd5e806a2b4a43f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_6bd5e806a2b4a43f = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_6bd5e806a2b4a43f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_6bd5e806a2b4a43f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_6bd5e806a2b4a43f(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_6bd5e806a2b4a43f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_6bd5e806a2b4a43f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_6bd5e806a2b4a43f(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_6bd5e806a2b4a43f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 256 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v17_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
                int32_t v21_lead = v17_lead + (v18_i0 * 16);
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 16; ++v19_i1) {
                  float v24_data = glb_m1[(v21_lead + (v19_i1 * 16))];
                  r0[(v18_i0 + v19_i1)] = v24_data;
                }
              }
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v27_i0 = 0; v27_i0 < 1; ++v27_i0) {
                int32_t v30_lead = v17_lead + (v27_i0 * 16);
                #pragma unroll
                for (int32_t v28_i1 = 0; v28_i1 < 16; ++v28_i1) {
                  float v33_data = glb_m2[(v30_lead + (v28_i1 * 16))];
                  r1[(v27_i0 + v28_i1)] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir2[16]{};
              float v37_data = r0[0];
              float v38_data = r1[0];
              float v41_data = ir2[0];
              ir2[0] = (v41_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v44_data = r1[1];
              float v47_data = ir2[1];
              ir2[1] = (v47_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v50_data = r1[2];
              float v53_data = ir2[2];
              ir2[2] = (v53_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v56_data = r1[3];
              float v59_data = ir2[3];
              ir2[3] = (v59_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v62_data = r1[4];
              float v65_data = ir2[4];
              ir2[4] = (v65_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v68_data = r1[5];
              float v71_data = ir2[5];
              ir2[5] = (v71_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v74_data = r1[6];
              float v77_data = ir2[6];
              ir2[6] = (v77_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v80_data = r1[7];
              float v83_data = ir2[7];
              ir2[7] = (v83_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v86_data = r1[8];
              float v89_data = ir2[8];
              ir2[8] = (v89_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v92_data = r1[9];
              float v95_data = ir2[9];
              ir2[9] = (v95_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v98_data = r1[10];
              float v101_data = ir2[10];
              ir2[10] = (v101_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v104_data = r1[11];
              float v107_data = ir2[11];
              ir2[11] = (v107_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v110_data = r1[12];
              float v113_data = ir2[12];
              ir2[12] = (v113_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v116_data = r1[13];
              float v119_data = ir2[13];
              ir2[13] = (v119_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v122_data = r1[14];
              float v125_data = ir2[14];
              ir2[14] = (v125_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v128_data = r1[15];
              float v131_data = ir2[15];
              ir2[15] = (v131_data + (v37_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v133_data = r0[1];
              float v137_data = ir2[0];
              ir2[0] = (v137_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v143_data = ir2[1];
              ir2[1] = (v143_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v149_data = ir2[2];
              ir2[2] = (v149_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v155_data = ir2[3];
              ir2[3] = (v155_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v161_data = ir2[4];
              ir2[4] = (v161_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v167_data = ir2[5];
              ir2[5] = (v167_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v173_data = ir2[6];
              ir2[6] = (v173_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v179_data = ir2[7];
              ir2[7] = (v179_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v185_data = ir2[8];
              ir2[8] = (v185_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v191_data = ir2[9];
              ir2[9] = (v191_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v197_data = ir2[10];
              ir2[10] = (v197_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v203_data = ir2[11];
              ir2[11] = (v203_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v209_data = ir2[12];
              ir2[12] = (v209_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v215_data = ir2[13];
              ir2[13] = (v215_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v221_data = ir2[14];
              ir2[14] = (v221_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v227_data = ir2[15];
              ir2[15] = (v227_data + (v133_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v229_data = r0[2];
              float v233_data = ir2[0];
              ir2[0] = (v233_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v239_data = ir2[1];
              ir2[1] = (v239_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v245_data = ir2[2];
              ir2[2] = (v245_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v251_data = ir2[3];
              ir2[3] = (v251_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v257_data = ir2[4];
              ir2[4] = (v257_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v263_data = ir2[5];
              ir2[5] = (v263_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v269_data = ir2[6];
              ir2[6] = (v269_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v275_data = ir2[7];
              ir2[7] = (v275_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v281_data = ir2[8];
              ir2[8] = (v281_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v287_data = ir2[9];
              ir2[9] = (v287_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v293_data = ir2[10];
              ir2[10] = (v293_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v299_data = ir2[11];
              ir2[11] = (v299_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v305_data = ir2[12];
              ir2[12] = (v305_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v311_data = ir2[13];
              ir2[13] = (v311_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v317_data = ir2[14];
              ir2[14] = (v317_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v323_data = ir2[15];
              ir2[15] = (v323_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v325_data = r0[3];
              float v329_data = ir2[0];
              ir2[0] = (v329_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v335_data = ir2[1];
              ir2[1] = (v335_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v341_data = ir2[2];
              ir2[2] = (v341_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v347_data = ir2[3];
              ir2[3] = (v347_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v353_data = ir2[4];
              ir2[4] = (v353_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v359_data = ir2[5];
              ir2[5] = (v359_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v365_data = ir2[6];
              ir2[6] = (v365_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v371_data = ir2[7];
              ir2[7] = (v371_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v377_data = ir2[8];
              ir2[8] = (v377_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v383_data = ir2[9];
              ir2[9] = (v383_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v389_data = ir2[10];
              ir2[10] = (v389_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v395_data = ir2[11];
              ir2[11] = (v395_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v401_data = ir2[12];
              ir2[12] = (v401_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v407_data = ir2[13];
              ir2[13] = (v407_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v413_data = ir2[14];
              ir2[14] = (v413_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v419_data = ir2[15];
              ir2[15] = (v419_data + (v325_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v421_data = r0[4];
              float v425_data = ir2[0];
              ir2[0] = (v425_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v431_data = ir2[1];
              ir2[1] = (v431_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v437_data = ir2[2];
              ir2[2] = (v437_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v443_data = ir2[3];
              ir2[3] = (v443_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v449_data = ir2[4];
              ir2[4] = (v449_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v455_data = ir2[5];
              ir2[5] = (v455_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v461_data = ir2[6];
              ir2[6] = (v461_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v467_data = ir2[7];
              ir2[7] = (v467_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v473_data = ir2[8];
              ir2[8] = (v473_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v479_data = ir2[9];
              ir2[9] = (v479_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v485_data = ir2[10];
              ir2[10] = (v485_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v491_data = ir2[11];
              ir2[11] = (v491_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v497_data = ir2[12];
              ir2[12] = (v497_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v503_data = ir2[13];
              ir2[13] = (v503_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v509_data = ir2[14];
              ir2[14] = (v509_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v515_data = ir2[15];
              ir2[15] = (v515_data + (v421_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v517_data = r0[5];
              float v521_data = ir2[0];
              ir2[0] = (v521_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v527_data = ir2[1];
              ir2[1] = (v527_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v533_data = ir2[2];
              ir2[2] = (v533_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v539_data = ir2[3];
              ir2[3] = (v539_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v545_data = ir2[4];
              ir2[4] = (v545_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v551_data = ir2[5];
              ir2[5] = (v551_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v557_data = ir2[6];
              ir2[6] = (v557_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v563_data = ir2[7];
              ir2[7] = (v563_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v569_data = ir2[8];
              ir2[8] = (v569_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v575_data = ir2[9];
              ir2[9] = (v575_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v581_data = ir2[10];
              ir2[10] = (v581_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v587_data = ir2[11];
              ir2[11] = (v587_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v593_data = ir2[12];
              ir2[12] = (v593_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v599_data = ir2[13];
              ir2[13] = (v599_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v605_data = ir2[14];
              ir2[14] = (v605_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v611_data = ir2[15];
              ir2[15] = (v611_data + (v517_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v613_data = r0[6];
              float v617_data = ir2[0];
              ir2[0] = (v617_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v623_data = ir2[1];
              ir2[1] = (v623_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v629_data = ir2[2];
              ir2[2] = (v629_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v635_data = ir2[3];
              ir2[3] = (v635_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v641_data = ir2[4];
              ir2[4] = (v641_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v647_data = ir2[5];
              ir2[5] = (v647_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v653_data = ir2[6];
              ir2[6] = (v653_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v659_data = ir2[7];
              ir2[7] = (v659_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v665_data = ir2[8];
              ir2[8] = (v665_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v671_data = ir2[9];
              ir2[9] = (v671_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v677_data = ir2[10];
              ir2[10] = (v677_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v683_data = ir2[11];
              ir2[11] = (v683_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v689_data = ir2[12];
              ir2[12] = (v689_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v695_data = ir2[13];
              ir2[13] = (v695_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v701_data = ir2[14];
              ir2[14] = (v701_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v707_data = ir2[15];
              ir2[15] = (v707_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v709_data = r0[7];
              float v713_data = ir2[0];
              ir2[0] = (v713_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v719_data = ir2[1];
              ir2[1] = (v719_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v725_data = ir2[2];
              ir2[2] = (v725_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v731_data = ir2[3];
              ir2[3] = (v731_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v737_data = ir2[4];
              ir2[4] = (v737_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v743_data = ir2[5];
              ir2[5] = (v743_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v749_data = ir2[6];
              ir2[6] = (v749_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v755_data = ir2[7];
              ir2[7] = (v755_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v761_data = ir2[8];
              ir2[8] = (v761_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v767_data = ir2[9];
              ir2[9] = (v767_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v773_data = ir2[10];
              ir2[10] = (v773_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v779_data = ir2[11];
              ir2[11] = (v779_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v785_data = ir2[12];
              ir2[12] = (v785_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v791_data = ir2[13];
              ir2[13] = (v791_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v797_data = ir2[14];
              ir2[14] = (v797_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v803_data = ir2[15];
              ir2[15] = (v803_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v805_data = r0[8];
              float v809_data = ir2[0];
              ir2[0] = (v809_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v815_data = ir2[1];
              ir2[1] = (v815_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v821_data = ir2[2];
              ir2[2] = (v821_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v827_data = ir2[3];
              ir2[3] = (v827_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v833_data = ir2[4];
              ir2[4] = (v833_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v839_data = ir2[5];
              ir2[5] = (v839_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v845_data = ir2[6];
              ir2[6] = (v845_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v851_data = ir2[7];
              ir2[7] = (v851_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v857_data = ir2[8];
              ir2[8] = (v857_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v863_data = ir2[9];
              ir2[9] = (v863_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v869_data = ir2[10];
              ir2[10] = (v869_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v875_data = ir2[11];
              ir2[11] = (v875_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v881_data = ir2[12];
              ir2[12] = (v881_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v887_data = ir2[13];
              ir2[13] = (v887_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v893_data = ir2[14];
              ir2[14] = (v893_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v899_data = ir2[15];
              ir2[15] = (v899_data + (v805_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v901_data = r0[9];
              float v905_data = ir2[0];
              ir2[0] = (v905_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v911_data = ir2[1];
              ir2[1] = (v911_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v917_data = ir2[2];
              ir2[2] = (v917_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v923_data = ir2[3];
              ir2[3] = (v923_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v929_data = ir2[4];
              ir2[4] = (v929_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v935_data = ir2[5];
              ir2[5] = (v935_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v941_data = ir2[6];
              ir2[6] = (v941_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v947_data = ir2[7];
              ir2[7] = (v947_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v953_data = ir2[8];
              ir2[8] = (v953_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v959_data = ir2[9];
              ir2[9] = (v959_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v965_data = ir2[10];
              ir2[10] = (v965_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v971_data = ir2[11];
              ir2[11] = (v971_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v977_data = ir2[12];
              ir2[12] = (v977_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v983_data = ir2[13];
              ir2[13] = (v983_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v989_data = ir2[14];
              ir2[14] = (v989_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v995_data = ir2[15];
              ir2[15] = (v995_data + (v901_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v997_data = r0[10];
              float v1001_data = ir2[0];
              ir2[0] = (v1001_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1007_data = ir2[1];
              ir2[1] = (v1007_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1013_data = ir2[2];
              ir2[2] = (v1013_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1019_data = ir2[3];
              ir2[3] = (v1019_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1025_data = ir2[4];
              ir2[4] = (v1025_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1031_data = ir2[5];
              ir2[5] = (v1031_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1037_data = ir2[6];
              ir2[6] = (v1037_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1043_data = ir2[7];
              ir2[7] = (v1043_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1049_data = ir2[8];
              ir2[8] = (v1049_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1055_data = ir2[9];
              ir2[9] = (v1055_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1061_data = ir2[10];
              ir2[10] = (v1061_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1067_data = ir2[11];
              ir2[11] = (v1067_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1073_data = ir2[12];
              ir2[12] = (v1073_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1079_data = ir2[13];
              ir2[13] = (v1079_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1085_data = ir2[14];
              ir2[14] = (v1085_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1091_data = ir2[15];
              ir2[15] = (v1091_data + (v997_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1093_data = r0[11];
              float v1097_data = ir2[0];
              ir2[0] = (v1097_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1103_data = ir2[1];
              ir2[1] = (v1103_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1109_data = ir2[2];
              ir2[2] = (v1109_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1115_data = ir2[3];
              ir2[3] = (v1115_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1121_data = ir2[4];
              ir2[4] = (v1121_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1127_data = ir2[5];
              ir2[5] = (v1127_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1133_data = ir2[6];
              ir2[6] = (v1133_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1139_data = ir2[7];
              ir2[7] = (v1139_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1145_data = ir2[8];
              ir2[8] = (v1145_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1151_data = ir2[9];
              ir2[9] = (v1151_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1157_data = ir2[10];
              ir2[10] = (v1157_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1163_data = ir2[11];
              ir2[11] = (v1163_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1169_data = ir2[12];
              ir2[12] = (v1169_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1175_data = ir2[13];
              ir2[13] = (v1175_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1181_data = ir2[14];
              ir2[14] = (v1181_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1187_data = ir2[15];
              ir2[15] = (v1187_data + (v1093_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1189_data = r0[12];
              float v1193_data = ir2[0];
              ir2[0] = (v1193_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1199_data = ir2[1];
              ir2[1] = (v1199_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1205_data = ir2[2];
              ir2[2] = (v1205_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1211_data = ir2[3];
              ir2[3] = (v1211_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1217_data = ir2[4];
              ir2[4] = (v1217_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1223_data = ir2[5];
              ir2[5] = (v1223_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1229_data = ir2[6];
              ir2[6] = (v1229_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1235_data = ir2[7];
              ir2[7] = (v1235_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1241_data = ir2[8];
              ir2[8] = (v1241_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1247_data = ir2[9];
              ir2[9] = (v1247_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1253_data = ir2[10];
              ir2[10] = (v1253_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1259_data = ir2[11];
              ir2[11] = (v1259_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1265_data = ir2[12];
              ir2[12] = (v1265_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1271_data = ir2[13];
              ir2[13] = (v1271_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1277_data = ir2[14];
              ir2[14] = (v1277_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1283_data = ir2[15];
              ir2[15] = (v1283_data + (v1189_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1285_data = r0[13];
              float v1289_data = ir2[0];
              ir2[0] = (v1289_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1295_data = ir2[1];
              ir2[1] = (v1295_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1301_data = ir2[2];
              ir2[2] = (v1301_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1307_data = ir2[3];
              ir2[3] = (v1307_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1313_data = ir2[4];
              ir2[4] = (v1313_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1319_data = ir2[5];
              ir2[5] = (v1319_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1325_data = ir2[6];
              ir2[6] = (v1325_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1331_data = ir2[7];
              ir2[7] = (v1331_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1337_data = ir2[8];
              ir2[8] = (v1337_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1343_data = ir2[9];
              ir2[9] = (v1343_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1349_data = ir2[10];
              ir2[10] = (v1349_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1355_data = ir2[11];
              ir2[11] = (v1355_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1361_data = ir2[12];
              ir2[12] = (v1361_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1367_data = ir2[13];
              ir2[13] = (v1367_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1373_data = ir2[14];
              ir2[14] = (v1373_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1379_data = ir2[15];
              ir2[15] = (v1379_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1381_data = r0[14];
              float v1385_data = ir2[0];
              ir2[0] = (v1385_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1391_data = ir2[1];
              ir2[1] = (v1391_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1397_data = ir2[2];
              ir2[2] = (v1397_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1403_data = ir2[3];
              ir2[3] = (v1403_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1409_data = ir2[4];
              ir2[4] = (v1409_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1415_data = ir2[5];
              ir2[5] = (v1415_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1421_data = ir2[6];
              ir2[6] = (v1421_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1427_data = ir2[7];
              ir2[7] = (v1427_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1433_data = ir2[8];
              ir2[8] = (v1433_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1439_data = ir2[9];
              ir2[9] = (v1439_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1445_data = ir2[10];
              ir2[10] = (v1445_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1451_data = ir2[11];
              ir2[11] = (v1451_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1457_data = ir2[12];
              ir2[12] = (v1457_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1463_data = ir2[13];
              ir2[13] = (v1463_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1469_data = ir2[14];
              ir2[14] = (v1469_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1475_data = ir2[15];
              ir2[15] = (v1475_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1477_data = r0[15];
              float v1481_data = ir2[0];
              ir2[0] = (v1481_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1487_data = ir2[1];
              ir2[1] = (v1487_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1493_data = ir2[2];
              ir2[2] = (v1493_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1499_data = ir2[3];
              ir2[3] = (v1499_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1505_data = ir2[4];
              ir2[4] = (v1505_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1511_data = ir2[5];
              ir2[5] = (v1511_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1517_data = ir2[6];
              ir2[6] = (v1517_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1523_data = ir2[7];
              ir2[7] = (v1523_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1529_data = ir2[8];
              ir2[8] = (v1529_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1535_data = ir2[9];
              ir2[9] = (v1535_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1541_data = ir2[10];
              ir2[10] = (v1541_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1547_data = ir2[11];
              ir2[11] = (v1547_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1553_data = ir2[12];
              ir2[12] = (v1553_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1559_data = ir2[13];
              ir2[13] = (v1559_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1565_data = ir2[14];
              ir2[14] = (v1565_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1571_data = ir2[15];
              ir2[15] = (v1571_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              #pragma unroll
              for (int32_t v1573_n0 = 0; v1573_n0 < 1; ++v1573_n0) {
                #pragma unroll
                for (int32_t v1574_n1 = 0; v1574_n1 < 16; ++v1574_n1) {
                  int32_t v1575_a = v1573_n0 + v1574_n1;
                  float v1576_data = ir2[v1575_a];
                  r2[v1575_a] = v1576_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1577_i0 = 0; v1577_i0 < 1; ++v1577_i0) {
                int32_t v1582_lead = v17_lead + (v1577_i0 * 16);
                #pragma unroll
                for (int32_t v1578_i1 = 0; v1578_i1 < 16; ++v1578_i1) {
                  float v1580_data = r2[(v1577_i0 + v1578_i1)];
                  glb_m0[(v1582_lead + (v1578_i1 * 16))] = v1580_data;
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

