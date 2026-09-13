// === base name ===
kernel_34ca99c53233ca82

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_34ca99c53233ca82 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_34ca99c53233ca82(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_34ca99c53233ca82(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_34ca99c53233ca82(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_34ca99c53233ca82(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_34ca99c53233ca82(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_34ca99c53233ca82(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_34ca99c53233ca82(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} none
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const float *const __restrict__ glb_m1 = &m1[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 256 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v17_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
                int32_t v21_lead = v17_lead + (v18_i0 * 16);
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 16; ++v19_i1) {
                  float v24_data = glb_m2[(v21_lead + (v19_i1 * 16))];
                  r0[(v18_i0 + v19_i1)] = v24_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m2););
              float r1[16]{};
              // r1 = +(glb_m1 * r0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir1[16]{};
              float v31_data = glb_m1[v17_lead];
              float v32_data = r0[0];
              float v35_data = ir1[0];
              ir1[0] = (v35_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v38_data = r0[1];
              float v41_data = ir1[1];
              ir1[1] = (v41_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v44_data = r0[2];
              float v47_data = ir1[2];
              ir1[2] = (v47_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v50_data = r0[3];
              float v53_data = ir1[3];
              ir1[3] = (v53_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v56_data = r0[4];
              float v59_data = ir1[4];
              ir1[4] = (v59_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v62_data = r0[5];
              float v65_data = ir1[5];
              ir1[5] = (v65_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v68_data = r0[6];
              float v71_data = ir1[6];
              ir1[6] = (v71_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v74_data = r0[7];
              float v77_data = ir1[7];
              ir1[7] = (v77_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v80_data = r0[8];
              float v83_data = ir1[8];
              ir1[8] = (v83_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v86_data = r0[9];
              float v89_data = ir1[9];
              ir1[9] = (v89_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v92_data = r0[10];
              float v95_data = ir1[10];
              ir1[10] = (v95_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v98_data = r0[11];
              float v101_data = ir1[11];
              ir1[11] = (v101_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v104_data = r0[12];
              float v107_data = ir1[12];
              ir1[12] = (v107_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v110_data = r0[13];
              float v113_data = ir1[13];
              ir1[13] = (v113_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v116_data = r0[14];
              float v119_data = ir1[14];
              ir1[14] = (v119_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v122_data = r0[15];
              float v125_data = ir1[15];
              ir1[15] = (v125_data + (v31_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v128_data = glb_m1[(v17_lead + 16)];
              float v132_data = ir1[0];
              ir1[0] = (v132_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v138_data = ir1[1];
              ir1[1] = (v138_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v144_data = ir1[2];
              ir1[2] = (v144_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v150_data = ir1[3];
              ir1[3] = (v150_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v156_data = ir1[4];
              ir1[4] = (v156_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v162_data = ir1[5];
              ir1[5] = (v162_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v168_data = ir1[6];
              ir1[6] = (v168_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v174_data = ir1[7];
              ir1[7] = (v174_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v180_data = ir1[8];
              ir1[8] = (v180_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v186_data = ir1[9];
              ir1[9] = (v186_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v192_data = ir1[10];
              ir1[10] = (v192_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v198_data = ir1[11];
              ir1[11] = (v198_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v204_data = ir1[12];
              ir1[12] = (v204_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v210_data = ir1[13];
              ir1[13] = (v210_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v216_data = ir1[14];
              ir1[14] = (v216_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v222_data = ir1[15];
              ir1[15] = (v222_data + (v128_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v225_data = glb_m1[(v17_lead + 32)];
              float v229_data = ir1[0];
              ir1[0] = (v229_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v235_data = ir1[1];
              ir1[1] = (v235_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v241_data = ir1[2];
              ir1[2] = (v241_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v247_data = ir1[3];
              ir1[3] = (v247_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v253_data = ir1[4];
              ir1[4] = (v253_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v259_data = ir1[5];
              ir1[5] = (v259_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v265_data = ir1[6];
              ir1[6] = (v265_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v271_data = ir1[7];
              ir1[7] = (v271_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v277_data = ir1[8];
              ir1[8] = (v277_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v283_data = ir1[9];
              ir1[9] = (v283_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v289_data = ir1[10];
              ir1[10] = (v289_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v295_data = ir1[11];
              ir1[11] = (v295_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v301_data = ir1[12];
              ir1[12] = (v301_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v307_data = ir1[13];
              ir1[13] = (v307_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v313_data = ir1[14];
              ir1[14] = (v313_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v319_data = ir1[15];
              ir1[15] = (v319_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v322_data = glb_m1[(v17_lead + 48)];
              float v326_data = ir1[0];
              ir1[0] = (v326_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v332_data = ir1[1];
              ir1[1] = (v332_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v338_data = ir1[2];
              ir1[2] = (v338_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v344_data = ir1[3];
              ir1[3] = (v344_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v350_data = ir1[4];
              ir1[4] = (v350_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v356_data = ir1[5];
              ir1[5] = (v356_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v362_data = ir1[6];
              ir1[6] = (v362_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v368_data = ir1[7];
              ir1[7] = (v368_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v374_data = ir1[8];
              ir1[8] = (v374_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v380_data = ir1[9];
              ir1[9] = (v380_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v386_data = ir1[10];
              ir1[10] = (v386_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v392_data = ir1[11];
              ir1[11] = (v392_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v398_data = ir1[12];
              ir1[12] = (v398_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v404_data = ir1[13];
              ir1[13] = (v404_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v410_data = ir1[14];
              ir1[14] = (v410_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v416_data = ir1[15];
              ir1[15] = (v416_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v419_data = glb_m1[(v17_lead + 64)];
              float v423_data = ir1[0];
              ir1[0] = (v423_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v429_data = ir1[1];
              ir1[1] = (v429_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v435_data = ir1[2];
              ir1[2] = (v435_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v441_data = ir1[3];
              ir1[3] = (v441_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v447_data = ir1[4];
              ir1[4] = (v447_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v453_data = ir1[5];
              ir1[5] = (v453_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v459_data = ir1[6];
              ir1[6] = (v459_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v465_data = ir1[7];
              ir1[7] = (v465_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v471_data = ir1[8];
              ir1[8] = (v471_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v477_data = ir1[9];
              ir1[9] = (v477_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v483_data = ir1[10];
              ir1[10] = (v483_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v489_data = ir1[11];
              ir1[11] = (v489_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v495_data = ir1[12];
              ir1[12] = (v495_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v501_data = ir1[13];
              ir1[13] = (v501_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v507_data = ir1[14];
              ir1[14] = (v507_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v513_data = ir1[15];
              ir1[15] = (v513_data + (v419_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v516_data = glb_m1[(v17_lead + 80)];
              float v520_data = ir1[0];
              ir1[0] = (v520_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v526_data = ir1[1];
              ir1[1] = (v526_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v532_data = ir1[2];
              ir1[2] = (v532_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v538_data = ir1[3];
              ir1[3] = (v538_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v544_data = ir1[4];
              ir1[4] = (v544_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v550_data = ir1[5];
              ir1[5] = (v550_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v556_data = ir1[6];
              ir1[6] = (v556_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v562_data = ir1[7];
              ir1[7] = (v562_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v568_data = ir1[8];
              ir1[8] = (v568_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v574_data = ir1[9];
              ir1[9] = (v574_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v580_data = ir1[10];
              ir1[10] = (v580_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v586_data = ir1[11];
              ir1[11] = (v586_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v592_data = ir1[12];
              ir1[12] = (v592_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v598_data = ir1[13];
              ir1[13] = (v598_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v604_data = ir1[14];
              ir1[14] = (v604_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v610_data = ir1[15];
              ir1[15] = (v610_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v613_data = glb_m1[(v17_lead + 96)];
              float v617_data = ir1[0];
              ir1[0] = (v617_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v623_data = ir1[1];
              ir1[1] = (v623_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v629_data = ir1[2];
              ir1[2] = (v629_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v635_data = ir1[3];
              ir1[3] = (v635_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v641_data = ir1[4];
              ir1[4] = (v641_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v647_data = ir1[5];
              ir1[5] = (v647_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v653_data = ir1[6];
              ir1[6] = (v653_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v659_data = ir1[7];
              ir1[7] = (v659_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v665_data = ir1[8];
              ir1[8] = (v665_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v671_data = ir1[9];
              ir1[9] = (v671_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v677_data = ir1[10];
              ir1[10] = (v677_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v683_data = ir1[11];
              ir1[11] = (v683_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v689_data = ir1[12];
              ir1[12] = (v689_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v695_data = ir1[13];
              ir1[13] = (v695_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v701_data = ir1[14];
              ir1[14] = (v701_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v707_data = ir1[15];
              ir1[15] = (v707_data + (v613_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v710_data = glb_m1[(v17_lead + 112)];
              float v714_data = ir1[0];
              ir1[0] = (v714_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v720_data = ir1[1];
              ir1[1] = (v720_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v726_data = ir1[2];
              ir1[2] = (v726_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v732_data = ir1[3];
              ir1[3] = (v732_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v738_data = ir1[4];
              ir1[4] = (v738_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v744_data = ir1[5];
              ir1[5] = (v744_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v750_data = ir1[6];
              ir1[6] = (v750_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v756_data = ir1[7];
              ir1[7] = (v756_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v762_data = ir1[8];
              ir1[8] = (v762_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v768_data = ir1[9];
              ir1[9] = (v768_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v774_data = ir1[10];
              ir1[10] = (v774_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v780_data = ir1[11];
              ir1[11] = (v780_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v786_data = ir1[12];
              ir1[12] = (v786_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v792_data = ir1[13];
              ir1[13] = (v792_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v798_data = ir1[14];
              ir1[14] = (v798_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v804_data = ir1[15];
              ir1[15] = (v804_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v807_data = glb_m1[(v17_lead + 128)];
              float v811_data = ir1[0];
              ir1[0] = (v811_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v817_data = ir1[1];
              ir1[1] = (v817_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v823_data = ir1[2];
              ir1[2] = (v823_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v829_data = ir1[3];
              ir1[3] = (v829_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v835_data = ir1[4];
              ir1[4] = (v835_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v841_data = ir1[5];
              ir1[5] = (v841_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v847_data = ir1[6];
              ir1[6] = (v847_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v853_data = ir1[7];
              ir1[7] = (v853_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v859_data = ir1[8];
              ir1[8] = (v859_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v865_data = ir1[9];
              ir1[9] = (v865_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v871_data = ir1[10];
              ir1[10] = (v871_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v877_data = ir1[11];
              ir1[11] = (v877_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v883_data = ir1[12];
              ir1[12] = (v883_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v889_data = ir1[13];
              ir1[13] = (v889_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v895_data = ir1[14];
              ir1[14] = (v895_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v901_data = ir1[15];
              ir1[15] = (v901_data + (v807_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v904_data = glb_m1[(v17_lead + 144)];
              float v908_data = ir1[0];
              ir1[0] = (v908_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v914_data = ir1[1];
              ir1[1] = (v914_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v920_data = ir1[2];
              ir1[2] = (v920_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v926_data = ir1[3];
              ir1[3] = (v926_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v932_data = ir1[4];
              ir1[4] = (v932_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v938_data = ir1[5];
              ir1[5] = (v938_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v944_data = ir1[6];
              ir1[6] = (v944_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v950_data = ir1[7];
              ir1[7] = (v950_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v956_data = ir1[8];
              ir1[8] = (v956_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v962_data = ir1[9];
              ir1[9] = (v962_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v968_data = ir1[10];
              ir1[10] = (v968_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v974_data = ir1[11];
              ir1[11] = (v974_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v980_data = ir1[12];
              ir1[12] = (v980_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v986_data = ir1[13];
              ir1[13] = (v986_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v992_data = ir1[14];
              ir1[14] = (v992_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v998_data = ir1[15];
              ir1[15] = (v998_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1001_data = glb_m1[(v17_lead + 160)];
              float v1005_data = ir1[0];
              ir1[0] = (v1005_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1011_data = ir1[1];
              ir1[1] = (v1011_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1017_data = ir1[2];
              ir1[2] = (v1017_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1023_data = ir1[3];
              ir1[3] = (v1023_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1029_data = ir1[4];
              ir1[4] = (v1029_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1035_data = ir1[5];
              ir1[5] = (v1035_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1041_data = ir1[6];
              ir1[6] = (v1041_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1047_data = ir1[7];
              ir1[7] = (v1047_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1053_data = ir1[8];
              ir1[8] = (v1053_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1059_data = ir1[9];
              ir1[9] = (v1059_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1065_data = ir1[10];
              ir1[10] = (v1065_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1071_data = ir1[11];
              ir1[11] = (v1071_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1077_data = ir1[12];
              ir1[12] = (v1077_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1083_data = ir1[13];
              ir1[13] = (v1083_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1089_data = ir1[14];
              ir1[14] = (v1089_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1095_data = ir1[15];
              ir1[15] = (v1095_data + (v1001_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1098_data = glb_m1[(v17_lead + 176)];
              float v1102_data = ir1[0];
              ir1[0] = (v1102_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1108_data = ir1[1];
              ir1[1] = (v1108_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1114_data = ir1[2];
              ir1[2] = (v1114_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1120_data = ir1[3];
              ir1[3] = (v1120_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1126_data = ir1[4];
              ir1[4] = (v1126_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1132_data = ir1[5];
              ir1[5] = (v1132_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1138_data = ir1[6];
              ir1[6] = (v1138_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1144_data = ir1[7];
              ir1[7] = (v1144_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1150_data = ir1[8];
              ir1[8] = (v1150_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1156_data = ir1[9];
              ir1[9] = (v1156_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1162_data = ir1[10];
              ir1[10] = (v1162_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1168_data = ir1[11];
              ir1[11] = (v1168_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1174_data = ir1[12];
              ir1[12] = (v1174_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1180_data = ir1[13];
              ir1[13] = (v1180_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1186_data = ir1[14];
              ir1[14] = (v1186_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1192_data = ir1[15];
              ir1[15] = (v1192_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1195_data = glb_m1[(v17_lead + 192)];
              float v1199_data = ir1[0];
              ir1[0] = (v1199_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1205_data = ir1[1];
              ir1[1] = (v1205_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1211_data = ir1[2];
              ir1[2] = (v1211_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1217_data = ir1[3];
              ir1[3] = (v1217_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1223_data = ir1[4];
              ir1[4] = (v1223_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1229_data = ir1[5];
              ir1[5] = (v1229_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1235_data = ir1[6];
              ir1[6] = (v1235_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1241_data = ir1[7];
              ir1[7] = (v1241_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1247_data = ir1[8];
              ir1[8] = (v1247_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1253_data = ir1[9];
              ir1[9] = (v1253_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1259_data = ir1[10];
              ir1[10] = (v1259_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1265_data = ir1[11];
              ir1[11] = (v1265_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1271_data = ir1[12];
              ir1[12] = (v1271_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1277_data = ir1[13];
              ir1[13] = (v1277_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1283_data = ir1[14];
              ir1[14] = (v1283_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1289_data = ir1[15];
              ir1[15] = (v1289_data + (v1195_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1292_data = glb_m1[(v17_lead + 208)];
              float v1296_data = ir1[0];
              ir1[0] = (v1296_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1302_data = ir1[1];
              ir1[1] = (v1302_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1308_data = ir1[2];
              ir1[2] = (v1308_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1314_data = ir1[3];
              ir1[3] = (v1314_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1320_data = ir1[4];
              ir1[4] = (v1320_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1326_data = ir1[5];
              ir1[5] = (v1326_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1332_data = ir1[6];
              ir1[6] = (v1332_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1338_data = ir1[7];
              ir1[7] = (v1338_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1344_data = ir1[8];
              ir1[8] = (v1344_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1350_data = ir1[9];
              ir1[9] = (v1350_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1356_data = ir1[10];
              ir1[10] = (v1356_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1362_data = ir1[11];
              ir1[11] = (v1362_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1368_data = ir1[12];
              ir1[12] = (v1368_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1374_data = ir1[13];
              ir1[13] = (v1374_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1380_data = ir1[14];
              ir1[14] = (v1380_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1386_data = ir1[15];
              ir1[15] = (v1386_data + (v1292_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1389_data = glb_m1[(v17_lead + 224)];
              float v1393_data = ir1[0];
              ir1[0] = (v1393_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1399_data = ir1[1];
              ir1[1] = (v1399_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1405_data = ir1[2];
              ir1[2] = (v1405_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1411_data = ir1[3];
              ir1[3] = (v1411_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1417_data = ir1[4];
              ir1[4] = (v1417_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1423_data = ir1[5];
              ir1[5] = (v1423_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1429_data = ir1[6];
              ir1[6] = (v1429_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1435_data = ir1[7];
              ir1[7] = (v1435_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1441_data = ir1[8];
              ir1[8] = (v1441_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1447_data = ir1[9];
              ir1[9] = (v1447_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1453_data = ir1[10];
              ir1[10] = (v1453_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1459_data = ir1[11];
              ir1[11] = (v1459_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1465_data = ir1[12];
              ir1[12] = (v1465_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1471_data = ir1[13];
              ir1[13] = (v1471_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1477_data = ir1[14];
              ir1[14] = (v1477_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1483_data = ir1[15];
              ir1[15] = (v1483_data + (v1389_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1486_data = glb_m1[(v17_lead + 240)];
              float v1490_data = ir1[0];
              ir1[0] = (v1490_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v32_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1496_data = ir1[1];
              ir1[1] = (v1496_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v38_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1502_data = ir1[2];
              ir1[2] = (v1502_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1508_data = ir1[3];
              ir1[3] = (v1508_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1514_data = ir1[4];
              ir1[4] = (v1514_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1520_data = ir1[5];
              ir1[5] = (v1520_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1526_data = ir1[6];
              ir1[6] = (v1526_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1532_data = ir1[7];
              ir1[7] = (v1532_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1538_data = ir1[8];
              ir1[8] = (v1538_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1544_data = ir1[9];
              ir1[9] = (v1544_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1550_data = ir1[10];
              ir1[10] = (v1550_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1556_data = ir1[11];
              ir1[11] = (v1556_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1562_data = ir1[12];
              ir1[12] = (v1562_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1568_data = ir1[13];
              ir1[13] = (v1568_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1574_data = ir1[14];
              ir1[14] = (v1574_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1580_data = ir1[15];
              ir1[15] = (v1580_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              #pragma unroll
              for (int32_t v1582_n0 = 0; v1582_n0 < 1; ++v1582_n0) {
                #pragma unroll
                for (int32_t v1583_n1 = 0; v1583_n1 < 16; ++v1583_n1) {
                  int32_t v1584_a = v1582_n0 + v1583_n1;
                  float v1585_data = ir1[v1584_a];
                  r1[v1584_a] = v1585_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v1586_i0 = 0; v1586_i0 < 1; ++v1586_i0) {
                int32_t v1591_lead = v17_lead + (v1586_i0 * 16);
                #pragma unroll
                for (int32_t v1587_i1 = 0; v1587_i1 < 16; ++v1587_i1) {
                  float v1589_data = r1[(v1586_i0 + v1587_i1)];
                  glb_m0[(v1591_lead + (v1587_i1 * 16))] = v1589_data;
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

