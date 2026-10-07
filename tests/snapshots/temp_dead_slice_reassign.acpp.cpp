// === base name ===
kernel_3bc97296b86d7eea

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3bc97296b86d7eea = {{16, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3bc97296b86d7eea(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3bc97296b86d7eea(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3bc97296b86d7eea(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2560 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_3bc97296b86d7eea(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3bc97296b86d7eea(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3bc97296b86d7eea(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3bc97296b86d7eea(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2560, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 6×12(6×12) {0..6}×{0..12} strided
        //   m1 12×12(12×12) {0..12}×{0..12} strided
        //   m2 6×12(6×12) {0..6}×{0..12} strided
        //   m3 6×12(6×12) {0..6}×{0..12} strided
        //   m4 12×12(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{6..12}×{0..12} = m0[i,k] × m1[k,j]
        //   t0[i,j] = m2[i,k] × m1[k,j]
        //   t0[i,j]@{6..12}×{0..12} = m3[i,j]
        //   m4[i,j] = t0[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"Y","bbox":[[0,0],[6,12]],"name":"m3","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[6,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[160 * item.get_local_id(1) + 0];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v8_batchId0 * 72 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v8_batchId0 * 144 + 0 + m4_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v24_lead = item.get_local_id(2) % 16;
              bool v25_g = v24_lead < 6;
              if (v25_g) {
                #pragma unroll
                for (int32_t v26_i1 = 0; v26_i1 < 12; ++v26_i1) {
                  float v31_data = glb_m0[(v24_lead + (v26_i1 * 6))];
                  r0[v26_i1] = v31_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              bool v34_g = v24_lead < 12;
              if (v34_g) {
                #pragma unroll
                for (int32_t v35_i1 = 0; v35_i1 < 12; ++v35_i1) {
                  float v40_data = glb_m1[(v24_lead + (v35_i1 * 12))];
                  r1[v35_i1] = v40_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m2);
              if (v25_g) {
                #pragma unroll
                for (int32_t v43_i1 = 0; v43_i1 < 12; ++v43_i1) {
                  float v48_data = glb_m2[(v24_lead + (v43_i1 * 6))];
                  r3[v43_i1] = v48_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[12]{};
              // r2 = +(r0 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              float v51_data = r0[0];
              float v52_data = r1[0];
              float v53_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v55_data = r2[0];
              r2[0] = (v55_data + (v51_data * v53_bc));
              float v58_data = r1[1];
              float v59_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v61_data = r2[1];
              r2[1] = (v61_data + (v51_data * v59_bc));
              float v64_data = r1[2];
              float v65_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v67_data = r2[2];
              r2[2] = (v67_data + (v51_data * v65_bc));
              float v70_data = r1[3];
              float v71_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v73_data = r2[3];
              r2[3] = (v73_data + (v51_data * v71_bc));
              float v76_data = r1[4];
              float v77_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v79_data = r2[4];
              r2[4] = (v79_data + (v51_data * v77_bc));
              float v82_data = r1[5];
              float v83_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v85_data = r2[5];
              r2[5] = (v85_data + (v51_data * v83_bc));
              float v88_data = r1[6];
              float v89_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v91_data = r2[6];
              r2[6] = (v91_data + (v51_data * v89_bc));
              float v94_data = r1[7];
              float v95_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v97_data = r2[7];
              r2[7] = (v97_data + (v51_data * v95_bc));
              float v100_data = r1[8];
              float v101_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v103_data = r2[8];
              r2[8] = (v103_data + (v51_data * v101_bc));
              float v106_data = r1[9];
              float v107_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v109_data = r2[9];
              r2[9] = (v109_data + (v51_data * v107_bc));
              float v112_data = r1[10];
              float v113_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v115_data = r2[10];
              r2[10] = (v115_data + (v51_data * v113_bc));
              float v118_data = r1[11];
              float v119_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v121_data = r2[11];
              r2[11] = (v121_data + (v51_data * v119_bc));
              float v123_data = r0[1];
              float v125_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v127_data = r2[0];
              r2[0] = (v127_data + (v123_data * v125_bc));
              float v131_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v133_data = r2[1];
              r2[1] = (v133_data + (v123_data * v131_bc));
              float v137_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v139_data = r2[2];
              r2[2] = (v139_data + (v123_data * v137_bc));
              float v143_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v145_data = r2[3];
              r2[3] = (v145_data + (v123_data * v143_bc));
              float v149_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v151_data = r2[4];
              r2[4] = (v151_data + (v123_data * v149_bc));
              float v155_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v157_data = r2[5];
              r2[5] = (v157_data + (v123_data * v155_bc));
              float v161_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v163_data = r2[6];
              r2[6] = (v163_data + (v123_data * v161_bc));
              float v167_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v169_data = r2[7];
              r2[7] = (v169_data + (v123_data * v167_bc));
              float v173_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v175_data = r2[8];
              r2[8] = (v175_data + (v123_data * v173_bc));
              float v179_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v181_data = r2[9];
              r2[9] = (v181_data + (v123_data * v179_bc));
              float v185_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v187_data = r2[10];
              r2[10] = (v187_data + (v123_data * v185_bc));
              float v191_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v193_data = r2[11];
              r2[11] = (v193_data + (v123_data * v191_bc));
              float v195_data = r0[2];
              float v197_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v199_data = r2[0];
              r2[0] = (v199_data + (v195_data * v197_bc));
              float v203_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v205_data = r2[1];
              r2[1] = (v205_data + (v195_data * v203_bc));
              float v209_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v211_data = r2[2];
              r2[2] = (v211_data + (v195_data * v209_bc));
              float v215_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v217_data = r2[3];
              r2[3] = (v217_data + (v195_data * v215_bc));
              float v221_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v223_data = r2[4];
              r2[4] = (v223_data + (v195_data * v221_bc));
              float v227_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v229_data = r2[5];
              r2[5] = (v229_data + (v195_data * v227_bc));
              float v233_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v235_data = r2[6];
              r2[6] = (v235_data + (v195_data * v233_bc));
              float v239_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v241_data = r2[7];
              r2[7] = (v241_data + (v195_data * v239_bc));
              float v245_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v247_data = r2[8];
              r2[8] = (v247_data + (v195_data * v245_bc));
              float v251_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v253_data = r2[9];
              r2[9] = (v253_data + (v195_data * v251_bc));
              float v257_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v259_data = r2[10];
              r2[10] = (v259_data + (v195_data * v257_bc));
              float v263_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v265_data = r2[11];
              r2[11] = (v265_data + (v195_data * v263_bc));
              float v267_data = r0[3];
              float v269_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v271_data = r2[0];
              r2[0] = (v271_data + (v267_data * v269_bc));
              float v275_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v277_data = r2[1];
              r2[1] = (v277_data + (v267_data * v275_bc));
              float v281_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v283_data = r2[2];
              r2[2] = (v283_data + (v267_data * v281_bc));
              float v287_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v289_data = r2[3];
              r2[3] = (v289_data + (v267_data * v287_bc));
              float v293_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v295_data = r2[4];
              r2[4] = (v295_data + (v267_data * v293_bc));
              float v299_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v301_data = r2[5];
              r2[5] = (v301_data + (v267_data * v299_bc));
              float v305_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v307_data = r2[6];
              r2[6] = (v307_data + (v267_data * v305_bc));
              float v311_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v313_data = r2[7];
              r2[7] = (v313_data + (v267_data * v311_bc));
              float v317_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v319_data = r2[8];
              r2[8] = (v319_data + (v267_data * v317_bc));
              float v323_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v325_data = r2[9];
              r2[9] = (v325_data + (v267_data * v323_bc));
              float v329_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v331_data = r2[10];
              r2[10] = (v331_data + (v267_data * v329_bc));
              float v335_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v337_data = r2[11];
              r2[11] = (v337_data + (v267_data * v335_bc));
              float v339_data = r0[4];
              float v341_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v343_data = r2[0];
              r2[0] = (v343_data + (v339_data * v341_bc));
              float v347_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v349_data = r2[1];
              r2[1] = (v349_data + (v339_data * v347_bc));
              float v353_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v355_data = r2[2];
              r2[2] = (v355_data + (v339_data * v353_bc));
              float v359_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v361_data = r2[3];
              r2[3] = (v361_data + (v339_data * v359_bc));
              float v365_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v367_data = r2[4];
              r2[4] = (v367_data + (v339_data * v365_bc));
              float v371_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v373_data = r2[5];
              r2[5] = (v373_data + (v339_data * v371_bc));
              float v377_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v379_data = r2[6];
              r2[6] = (v379_data + (v339_data * v377_bc));
              float v383_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v385_data = r2[7];
              r2[7] = (v385_data + (v339_data * v383_bc));
              float v389_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v391_data = r2[8];
              r2[8] = (v391_data + (v339_data * v389_bc));
              float v395_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v397_data = r2[9];
              r2[9] = (v397_data + (v339_data * v395_bc));
              float v401_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v403_data = r2[10];
              r2[10] = (v403_data + (v339_data * v401_bc));
              float v407_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v409_data = r2[11];
              r2[11] = (v409_data + (v339_data * v407_bc));
              float v411_data = r0[5];
              float v413_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v415_data = r2[0];
              r2[0] = (v415_data + (v411_data * v413_bc));
              float v419_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v421_data = r2[1];
              r2[1] = (v421_data + (v411_data * v419_bc));
              float v425_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v427_data = r2[2];
              r2[2] = (v427_data + (v411_data * v425_bc));
              float v431_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v433_data = r2[3];
              r2[3] = (v433_data + (v411_data * v431_bc));
              float v437_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v439_data = r2[4];
              r2[4] = (v439_data + (v411_data * v437_bc));
              float v443_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v445_data = r2[5];
              r2[5] = (v445_data + (v411_data * v443_bc));
              float v449_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v451_data = r2[6];
              r2[6] = (v451_data + (v411_data * v449_bc));
              float v455_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v457_data = r2[7];
              r2[7] = (v457_data + (v411_data * v455_bc));
              float v461_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v463_data = r2[8];
              r2[8] = (v463_data + (v411_data * v461_bc));
              float v467_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v469_data = r2[9];
              r2[9] = (v469_data + (v411_data * v467_bc));
              float v473_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v475_data = r2[10];
              r2[10] = (v475_data + (v411_data * v473_bc));
              float v479_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v481_data = r2[11];
              r2[11] = (v481_data + (v411_data * v479_bc));
              float v483_data = r0[6];
              float v485_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v487_data = r2[0];
              r2[0] = (v487_data + (v483_data * v485_bc));
              float v491_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v493_data = r2[1];
              r2[1] = (v493_data + (v483_data * v491_bc));
              float v497_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v499_data = r2[2];
              r2[2] = (v499_data + (v483_data * v497_bc));
              float v503_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v505_data = r2[3];
              r2[3] = (v505_data + (v483_data * v503_bc));
              float v509_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v511_data = r2[4];
              r2[4] = (v511_data + (v483_data * v509_bc));
              float v515_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v517_data = r2[5];
              r2[5] = (v517_data + (v483_data * v515_bc));
              float v521_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v523_data = r2[6];
              r2[6] = (v523_data + (v483_data * v521_bc));
              float v527_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v529_data = r2[7];
              r2[7] = (v529_data + (v483_data * v527_bc));
              float v533_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v535_data = r2[8];
              r2[8] = (v535_data + (v483_data * v533_bc));
              float v539_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v541_data = r2[9];
              r2[9] = (v541_data + (v483_data * v539_bc));
              float v545_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v547_data = r2[10];
              r2[10] = (v547_data + (v483_data * v545_bc));
              float v551_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v553_data = r2[11];
              r2[11] = (v553_data + (v483_data * v551_bc));
              float v555_data = r0[7];
              float v557_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v559_data = r2[0];
              r2[0] = (v559_data + (v555_data * v557_bc));
              float v563_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v565_data = r2[1];
              r2[1] = (v565_data + (v555_data * v563_bc));
              float v569_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v571_data = r2[2];
              r2[2] = (v571_data + (v555_data * v569_bc));
              float v575_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v577_data = r2[3];
              r2[3] = (v577_data + (v555_data * v575_bc));
              float v581_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v583_data = r2[4];
              r2[4] = (v583_data + (v555_data * v581_bc));
              float v587_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v589_data = r2[5];
              r2[5] = (v589_data + (v555_data * v587_bc));
              float v593_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v595_data = r2[6];
              r2[6] = (v595_data + (v555_data * v593_bc));
              float v599_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v601_data = r2[7];
              r2[7] = (v601_data + (v555_data * v599_bc));
              float v605_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v607_data = r2[8];
              r2[8] = (v607_data + (v555_data * v605_bc));
              float v611_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v613_data = r2[9];
              r2[9] = (v613_data + (v555_data * v611_bc));
              float v617_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v619_data = r2[10];
              r2[10] = (v619_data + (v555_data * v617_bc));
              float v623_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v625_data = r2[11];
              r2[11] = (v625_data + (v555_data * v623_bc));
              float v627_data = r0[8];
              float v629_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v631_data = r2[0];
              r2[0] = (v631_data + (v627_data * v629_bc));
              float v635_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v637_data = r2[1];
              r2[1] = (v637_data + (v627_data * v635_bc));
              float v641_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v643_data = r2[2];
              r2[2] = (v643_data + (v627_data * v641_bc));
              float v647_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v649_data = r2[3];
              r2[3] = (v649_data + (v627_data * v647_bc));
              float v653_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v655_data = r2[4];
              r2[4] = (v655_data + (v627_data * v653_bc));
              float v659_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v661_data = r2[5];
              r2[5] = (v661_data + (v627_data * v659_bc));
              float v665_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v667_data = r2[6];
              r2[6] = (v667_data + (v627_data * v665_bc));
              float v671_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v673_data = r2[7];
              r2[7] = (v673_data + (v627_data * v671_bc));
              float v677_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v679_data = r2[8];
              r2[8] = (v679_data + (v627_data * v677_bc));
              float v683_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v685_data = r2[9];
              r2[9] = (v685_data + (v627_data * v683_bc));
              float v689_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v691_data = r2[10];
              r2[10] = (v691_data + (v627_data * v689_bc));
              float v695_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v697_data = r2[11];
              r2[11] = (v697_data + (v627_data * v695_bc));
              float v699_data = r0[9];
              float v701_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v703_data = r2[0];
              r2[0] = (v703_data + (v699_data * v701_bc));
              float v707_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v709_data = r2[1];
              r2[1] = (v709_data + (v699_data * v707_bc));
              float v713_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v715_data = r2[2];
              r2[2] = (v715_data + (v699_data * v713_bc));
              float v719_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v721_data = r2[3];
              r2[3] = (v721_data + (v699_data * v719_bc));
              float v725_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v727_data = r2[4];
              r2[4] = (v727_data + (v699_data * v725_bc));
              float v731_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v733_data = r2[5];
              r2[5] = (v733_data + (v699_data * v731_bc));
              float v737_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v739_data = r2[6];
              r2[6] = (v739_data + (v699_data * v737_bc));
              float v743_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v745_data = r2[7];
              r2[7] = (v745_data + (v699_data * v743_bc));
              float v749_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v751_data = r2[8];
              r2[8] = (v751_data + (v699_data * v749_bc));
              float v755_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v757_data = r2[9];
              r2[9] = (v757_data + (v699_data * v755_bc));
              float v761_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v763_data = r2[10];
              r2[10] = (v763_data + (v699_data * v761_bc));
              float v767_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v769_data = r2[11];
              r2[11] = (v769_data + (v699_data * v767_bc));
              float v771_data = r0[10];
              float v773_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v775_data = r2[0];
              r2[0] = (v775_data + (v771_data * v773_bc));
              float v779_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v781_data = r2[1];
              r2[1] = (v781_data + (v771_data * v779_bc));
              float v785_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v787_data = r2[2];
              r2[2] = (v787_data + (v771_data * v785_bc));
              float v791_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v793_data = r2[3];
              r2[3] = (v793_data + (v771_data * v791_bc));
              float v797_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v799_data = r2[4];
              r2[4] = (v799_data + (v771_data * v797_bc));
              float v803_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v805_data = r2[5];
              r2[5] = (v805_data + (v771_data * v803_bc));
              float v809_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v811_data = r2[6];
              r2[6] = (v811_data + (v771_data * v809_bc));
              float v815_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v817_data = r2[7];
              r2[7] = (v817_data + (v771_data * v815_bc));
              float v821_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v823_data = r2[8];
              r2[8] = (v823_data + (v771_data * v821_bc));
              float v827_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v829_data = r2[9];
              r2[9] = (v829_data + (v771_data * v827_bc));
              float v833_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v835_data = r2[10];
              r2[10] = (v835_data + (v771_data * v833_bc));
              float v839_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v841_data = r2[11];
              r2[11] = (v841_data + (v771_data * v839_bc));
              float v843_data = r0[11];
              float v845_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v847_data = r2[0];
              r2[0] = (v847_data + (v843_data * v845_bc));
              float v851_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v853_data = r2[1];
              r2[1] = (v853_data + (v843_data * v851_bc));
              float v857_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v859_data = r2[2];
              r2[2] = (v859_data + (v843_data * v857_bc));
              float v863_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v865_data = r2[3];
              r2[3] = (v865_data + (v843_data * v863_bc));
              float v869_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v871_data = r2[4];
              r2[4] = (v871_data + (v843_data * v869_bc));
              float v875_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v877_data = r2[5];
              r2[5] = (v877_data + (v843_data * v875_bc));
              float v881_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v883_data = r2[6];
              r2[6] = (v883_data + (v843_data * v881_bc));
              float v887_bc = sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v889_data = r2[7];
              r2[7] = (v889_data + (v843_data * v887_bc));
              float v893_bc = sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v895_data = r2[8];
              r2[8] = (v895_data + (v843_data * v893_bc));
              float v899_bc = sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v901_data = r2[9];
              r2[9] = (v901_data + (v843_data * v899_bc));
              float v905_bc = sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v907_data = r2[10];
              r2[10] = (v907_data + (v843_data * v905_bc));
              float v911_bc = sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v913_data = r2[11];
              r2[11] = (v913_data + (v843_data * v911_bc));
              // s0 = store{r>s}(localShrMem0, r2);
              if (v25_g) {
                int32_t v920_off = v24_lead + 6;
                #pragma unroll
                for (int32_t v915_i1 = 0; v915_i1 < 12; ++v915_i1) {
                  float v917_data = r2[v915_i1];
                  int32_t v922_a = v920_off + (v915_i1 * 12);
                  s0[(v922_a ^ ((v922_a >> 4) & 15))] = v917_data;
                }
              }
              float r5[12]{};
              // r5 = load{g>r}(glb_m3);
              if (v25_g) {
                #pragma unroll
                for (int32_t v927_i1 = 0; v927_i1 < 12; ++v927_i1) {
                  float v932_data = glb_m3[(v24_lead + (v927_i1 * 6))];
                  r5[v927_i1] = v932_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[12]{};
              // ir4 = +(r3 * r1)
              // [(0, 6), (0, 12)] [(0, 12)]
              float ir4[12]{};
              float v936_data = r3[0];
              float v940_data = ir4[0];
              ir4[0] = (v940_data + (v936_data * v53_bc));
              float v946_data = ir4[1];
              ir4[1] = (v946_data + (v936_data * v59_bc));
              float v952_data = ir4[2];
              ir4[2] = (v952_data + (v936_data * v65_bc));
              float v958_data = ir4[3];
              ir4[3] = (v958_data + (v936_data * v71_bc));
              float v964_data = ir4[4];
              ir4[4] = (v964_data + (v936_data * v77_bc));
              float v970_data = ir4[5];
              ir4[5] = (v970_data + (v936_data * v83_bc));
              float v976_data = ir4[6];
              ir4[6] = (v976_data + (v936_data * v89_bc));
              float v982_data = ir4[7];
              ir4[7] = (v982_data + (v936_data * v95_bc));
              float v988_data = ir4[8];
              ir4[8] = (v988_data + (v936_data * v101_bc));
              float v994_data = ir4[9];
              ir4[9] = (v994_data + (v936_data * v107_bc));
              float v1000_data = ir4[10];
              ir4[10] = (v1000_data + (v936_data * v113_bc));
              float v1006_data = ir4[11];
              ir4[11] = (v1006_data + (v936_data * v119_bc));
              float v1008_data = r3[1];
              float v1012_data = ir4[0];
              ir4[0] = (v1012_data + (v1008_data * v125_bc));
              float v1018_data = ir4[1];
              ir4[1] = (v1018_data + (v1008_data * v131_bc));
              float v1024_data = ir4[2];
              ir4[2] = (v1024_data + (v1008_data * v137_bc));
              float v1030_data = ir4[3];
              ir4[3] = (v1030_data + (v1008_data * v143_bc));
              float v1036_data = ir4[4];
              ir4[4] = (v1036_data + (v1008_data * v149_bc));
              float v1042_data = ir4[5];
              ir4[5] = (v1042_data + (v1008_data * v155_bc));
              float v1048_data = ir4[6];
              ir4[6] = (v1048_data + (v1008_data * v161_bc));
              float v1054_data = ir4[7];
              ir4[7] = (v1054_data + (v1008_data * v167_bc));
              float v1060_data = ir4[8];
              ir4[8] = (v1060_data + (v1008_data * v173_bc));
              float v1066_data = ir4[9];
              ir4[9] = (v1066_data + (v1008_data * v179_bc));
              float v1072_data = ir4[10];
              ir4[10] = (v1072_data + (v1008_data * v185_bc));
              float v1078_data = ir4[11];
              ir4[11] = (v1078_data + (v1008_data * v191_bc));
              float v1080_data = r3[2];
              float v1084_data = ir4[0];
              ir4[0] = (v1084_data + (v1080_data * v197_bc));
              float v1090_data = ir4[1];
              ir4[1] = (v1090_data + (v1080_data * v203_bc));
              float v1096_data = ir4[2];
              ir4[2] = (v1096_data + (v1080_data * v209_bc));
              float v1102_data = ir4[3];
              ir4[3] = (v1102_data + (v1080_data * v215_bc));
              float v1108_data = ir4[4];
              ir4[4] = (v1108_data + (v1080_data * v221_bc));
              float v1114_data = ir4[5];
              ir4[5] = (v1114_data + (v1080_data * v227_bc));
              float v1120_data = ir4[6];
              ir4[6] = (v1120_data + (v1080_data * v233_bc));
              float v1126_data = ir4[7];
              ir4[7] = (v1126_data + (v1080_data * v239_bc));
              float v1132_data = ir4[8];
              ir4[8] = (v1132_data + (v1080_data * v245_bc));
              float v1138_data = ir4[9];
              ir4[9] = (v1138_data + (v1080_data * v251_bc));
              float v1144_data = ir4[10];
              ir4[10] = (v1144_data + (v1080_data * v257_bc));
              float v1150_data = ir4[11];
              ir4[11] = (v1150_data + (v1080_data * v263_bc));
              float v1152_data = r3[3];
              float v1156_data = ir4[0];
              ir4[0] = (v1156_data + (v1152_data * v269_bc));
              float v1162_data = ir4[1];
              ir4[1] = (v1162_data + (v1152_data * v275_bc));
              float v1168_data = ir4[2];
              ir4[2] = (v1168_data + (v1152_data * v281_bc));
              float v1174_data = ir4[3];
              ir4[3] = (v1174_data + (v1152_data * v287_bc));
              float v1180_data = ir4[4];
              ir4[4] = (v1180_data + (v1152_data * v293_bc));
              float v1186_data = ir4[5];
              ir4[5] = (v1186_data + (v1152_data * v299_bc));
              float v1192_data = ir4[6];
              ir4[6] = (v1192_data + (v1152_data * v305_bc));
              float v1198_data = ir4[7];
              ir4[7] = (v1198_data + (v1152_data * v311_bc));
              float v1204_data = ir4[8];
              ir4[8] = (v1204_data + (v1152_data * v317_bc));
              float v1210_data = ir4[9];
              ir4[9] = (v1210_data + (v1152_data * v323_bc));
              float v1216_data = ir4[10];
              ir4[10] = (v1216_data + (v1152_data * v329_bc));
              float v1222_data = ir4[11];
              ir4[11] = (v1222_data + (v1152_data * v335_bc));
              float v1224_data = r3[4];
              float v1228_data = ir4[0];
              ir4[0] = (v1228_data + (v1224_data * v341_bc));
              float v1234_data = ir4[1];
              ir4[1] = (v1234_data + (v1224_data * v347_bc));
              float v1240_data = ir4[2];
              ir4[2] = (v1240_data + (v1224_data * v353_bc));
              float v1246_data = ir4[3];
              ir4[3] = (v1246_data + (v1224_data * v359_bc));
              float v1252_data = ir4[4];
              ir4[4] = (v1252_data + (v1224_data * v365_bc));
              float v1258_data = ir4[5];
              ir4[5] = (v1258_data + (v1224_data * v371_bc));
              float v1264_data = ir4[6];
              ir4[6] = (v1264_data + (v1224_data * v377_bc));
              float v1270_data = ir4[7];
              ir4[7] = (v1270_data + (v1224_data * v383_bc));
              float v1276_data = ir4[8];
              ir4[8] = (v1276_data + (v1224_data * v389_bc));
              float v1282_data = ir4[9];
              ir4[9] = (v1282_data + (v1224_data * v395_bc));
              float v1288_data = ir4[10];
              ir4[10] = (v1288_data + (v1224_data * v401_bc));
              float v1294_data = ir4[11];
              ir4[11] = (v1294_data + (v1224_data * v407_bc));
              float v1296_data = r3[5];
              float v1300_data = ir4[0];
              ir4[0] = (v1300_data + (v1296_data * v413_bc));
              float v1306_data = ir4[1];
              ir4[1] = (v1306_data + (v1296_data * v419_bc));
              float v1312_data = ir4[2];
              ir4[2] = (v1312_data + (v1296_data * v425_bc));
              float v1318_data = ir4[3];
              ir4[3] = (v1318_data + (v1296_data * v431_bc));
              float v1324_data = ir4[4];
              ir4[4] = (v1324_data + (v1296_data * v437_bc));
              float v1330_data = ir4[5];
              ir4[5] = (v1330_data + (v1296_data * v443_bc));
              float v1336_data = ir4[6];
              ir4[6] = (v1336_data + (v1296_data * v449_bc));
              float v1342_data = ir4[7];
              ir4[7] = (v1342_data + (v1296_data * v455_bc));
              float v1348_data = ir4[8];
              ir4[8] = (v1348_data + (v1296_data * v461_bc));
              float v1354_data = ir4[9];
              ir4[9] = (v1354_data + (v1296_data * v467_bc));
              float v1360_data = ir4[10];
              ir4[10] = (v1360_data + (v1296_data * v473_bc));
              float v1366_data = ir4[11];
              ir4[11] = (v1366_data + (v1296_data * v479_bc));
              float v1368_data = r3[6];
              float v1372_data = ir4[0];
              ir4[0] = (v1372_data + (v1368_data * v485_bc));
              float v1378_data = ir4[1];
              ir4[1] = (v1378_data + (v1368_data * v491_bc));
              float v1384_data = ir4[2];
              ir4[2] = (v1384_data + (v1368_data * v497_bc));
              float v1390_data = ir4[3];
              ir4[3] = (v1390_data + (v1368_data * v503_bc));
              float v1396_data = ir4[4];
              ir4[4] = (v1396_data + (v1368_data * v509_bc));
              float v1402_data = ir4[5];
              ir4[5] = (v1402_data + (v1368_data * v515_bc));
              float v1408_data = ir4[6];
              ir4[6] = (v1408_data + (v1368_data * v521_bc));
              float v1414_data = ir4[7];
              ir4[7] = (v1414_data + (v1368_data * v527_bc));
              float v1420_data = ir4[8];
              ir4[8] = (v1420_data + (v1368_data * v533_bc));
              float v1426_data = ir4[9];
              ir4[9] = (v1426_data + (v1368_data * v539_bc));
              float v1432_data = ir4[10];
              ir4[10] = (v1432_data + (v1368_data * v545_bc));
              float v1438_data = ir4[11];
              ir4[11] = (v1438_data + (v1368_data * v551_bc));
              float v1440_data = r3[7];
              float v1444_data = ir4[0];
              ir4[0] = (v1444_data + (v1440_data * v557_bc));
              float v1450_data = ir4[1];
              ir4[1] = (v1450_data + (v1440_data * v563_bc));
              float v1456_data = ir4[2];
              ir4[2] = (v1456_data + (v1440_data * v569_bc));
              float v1462_data = ir4[3];
              ir4[3] = (v1462_data + (v1440_data * v575_bc));
              float v1468_data = ir4[4];
              ir4[4] = (v1468_data + (v1440_data * v581_bc));
              float v1474_data = ir4[5];
              ir4[5] = (v1474_data + (v1440_data * v587_bc));
              float v1480_data = ir4[6];
              ir4[6] = (v1480_data + (v1440_data * v593_bc));
              float v1486_data = ir4[7];
              ir4[7] = (v1486_data + (v1440_data * v599_bc));
              float v1492_data = ir4[8];
              ir4[8] = (v1492_data + (v1440_data * v605_bc));
              float v1498_data = ir4[9];
              ir4[9] = (v1498_data + (v1440_data * v611_bc));
              float v1504_data = ir4[10];
              ir4[10] = (v1504_data + (v1440_data * v617_bc));
              float v1510_data = ir4[11];
              ir4[11] = (v1510_data + (v1440_data * v623_bc));
              float v1512_data = r3[8];
              float v1516_data = ir4[0];
              ir4[0] = (v1516_data + (v1512_data * v629_bc));
              float v1522_data = ir4[1];
              ir4[1] = (v1522_data + (v1512_data * v635_bc));
              float v1528_data = ir4[2];
              ir4[2] = (v1528_data + (v1512_data * v641_bc));
              float v1534_data = ir4[3];
              ir4[3] = (v1534_data + (v1512_data * v647_bc));
              float v1540_data = ir4[4];
              ir4[4] = (v1540_data + (v1512_data * v653_bc));
              float v1546_data = ir4[5];
              ir4[5] = (v1546_data + (v1512_data * v659_bc));
              float v1552_data = ir4[6];
              ir4[6] = (v1552_data + (v1512_data * v665_bc));
              float v1558_data = ir4[7];
              ir4[7] = (v1558_data + (v1512_data * v671_bc));
              float v1564_data = ir4[8];
              ir4[8] = (v1564_data + (v1512_data * v677_bc));
              float v1570_data = ir4[9];
              ir4[9] = (v1570_data + (v1512_data * v683_bc));
              float v1576_data = ir4[10];
              ir4[10] = (v1576_data + (v1512_data * v689_bc));
              float v1582_data = ir4[11];
              ir4[11] = (v1582_data + (v1512_data * v695_bc));
              float v1584_data = r3[9];
              float v1588_data = ir4[0];
              ir4[0] = (v1588_data + (v1584_data * v701_bc));
              float v1594_data = ir4[1];
              ir4[1] = (v1594_data + (v1584_data * v707_bc));
              float v1600_data = ir4[2];
              ir4[2] = (v1600_data + (v1584_data * v713_bc));
              float v1606_data = ir4[3];
              ir4[3] = (v1606_data + (v1584_data * v719_bc));
              float v1612_data = ir4[4];
              ir4[4] = (v1612_data + (v1584_data * v725_bc));
              float v1618_data = ir4[5];
              ir4[5] = (v1618_data + (v1584_data * v731_bc));
              float v1624_data = ir4[6];
              ir4[6] = (v1624_data + (v1584_data * v737_bc));
              float v1630_data = ir4[7];
              ir4[7] = (v1630_data + (v1584_data * v743_bc));
              float v1636_data = ir4[8];
              ir4[8] = (v1636_data + (v1584_data * v749_bc));
              float v1642_data = ir4[9];
              ir4[9] = (v1642_data + (v1584_data * v755_bc));
              float v1648_data = ir4[10];
              ir4[10] = (v1648_data + (v1584_data * v761_bc));
              float v1654_data = ir4[11];
              ir4[11] = (v1654_data + (v1584_data * v767_bc));
              float v1656_data = r3[10];
              float v1660_data = ir4[0];
              ir4[0] = (v1660_data + (v1656_data * v773_bc));
              float v1666_data = ir4[1];
              ir4[1] = (v1666_data + (v1656_data * v779_bc));
              float v1672_data = ir4[2];
              ir4[2] = (v1672_data + (v1656_data * v785_bc));
              float v1678_data = ir4[3];
              ir4[3] = (v1678_data + (v1656_data * v791_bc));
              float v1684_data = ir4[4];
              ir4[4] = (v1684_data + (v1656_data * v797_bc));
              float v1690_data = ir4[5];
              ir4[5] = (v1690_data + (v1656_data * v803_bc));
              float v1696_data = ir4[6];
              ir4[6] = (v1696_data + (v1656_data * v809_bc));
              float v1702_data = ir4[7];
              ir4[7] = (v1702_data + (v1656_data * v815_bc));
              float v1708_data = ir4[8];
              ir4[8] = (v1708_data + (v1656_data * v821_bc));
              float v1714_data = ir4[9];
              ir4[9] = (v1714_data + (v1656_data * v827_bc));
              float v1720_data = ir4[10];
              ir4[10] = (v1720_data + (v1656_data * v833_bc));
              float v1726_data = ir4[11];
              ir4[11] = (v1726_data + (v1656_data * v839_bc));
              float v1728_data = r3[11];
              float v1732_data = ir4[0];
              ir4[0] = (v1732_data + (v1728_data * v845_bc));
              float v1738_data = ir4[1];
              ir4[1] = (v1738_data + (v1728_data * v851_bc));
              float v1744_data = ir4[2];
              ir4[2] = (v1744_data + (v1728_data * v857_bc));
              float v1750_data = ir4[3];
              ir4[3] = (v1750_data + (v1728_data * v863_bc));
              float v1756_data = ir4[4];
              ir4[4] = (v1756_data + (v1728_data * v869_bc));
              float v1762_data = ir4[5];
              ir4[5] = (v1762_data + (v1728_data * v875_bc));
              float v1768_data = ir4[6];
              ir4[6] = (v1768_data + (v1728_data * v881_bc));
              float v1774_data = ir4[7];
              ir4[7] = (v1774_data + (v1728_data * v887_bc));
              float v1780_data = ir4[8];
              ir4[8] = (v1780_data + (v1728_data * v893_bc));
              float v1786_data = ir4[9];
              ir4[9] = (v1786_data + (v1728_data * v899_bc));
              float v1792_data = ir4[10];
              ir4[10] = (v1792_data + (v1728_data * v905_bc));
              float v1798_data = ir4[11];
              ir4[11] = (v1798_data + (v1728_data * v911_bc));
              // r4 = ir4
              if (v25_g) {
                #pragma unroll
                for (int32_t v1800_n1 = 0; v1800_n1 < 12; ++v1800_n1) {
                  float v1802_data = ir4[v1800_n1];
                  r4[v1800_n1] = v1802_data;
                }
              }
              // s0 = store{r>s, clear}(localShrMem0, r4);
              sycl::group_barrier(item.get_sub_group());
              if ((v24_lead >= 6) && v34_g) {
                #pragma unroll
                for (int32_t v1805_z1 = 0; v1805_z1 < 12; ++v1805_z1) {
                  int32_t v1810_a = v24_lead + (v1805_z1 * 12);
                  s0[(v1810_a ^ ((v1810_a >> 4) & 15))] = 0.0f;
                }
              }
              if (v25_g) {
                #pragma unroll
                for (int32_t v1814_i1 = 0; v1814_i1 < 12; ++v1814_i1) {
                  float v1816_data = r4[v1814_i1];
                  int32_t v1820_a = v24_lead + (v1814_i1 * 12);
                  s0[(v1820_a ^ ((v1820_a >> 4) & 15))] = v1816_data;
                }
              }
              // wait(r5 = load{g>r}(glb_m3););
              float r6[12]{};
              // ir6 = +(r5)
              // [(0, 6), (0, 12)] []
              float ir6[12]{};
              float v1826_data = r5[0];
              float v1827_data = ir6[0];
              ir6[0] = (v1827_data + v1826_data);
              float v1829_data = r5[1];
              float v1830_data = ir6[1];
              ir6[1] = (v1830_data + v1829_data);
              float v1832_data = r5[2];
              float v1833_data = ir6[2];
              ir6[2] = (v1833_data + v1832_data);
              float v1835_data = r5[3];
              float v1836_data = ir6[3];
              ir6[3] = (v1836_data + v1835_data);
              float v1838_data = r5[4];
              float v1839_data = ir6[4];
              ir6[4] = (v1839_data + v1838_data);
              float v1841_data = r5[5];
              float v1842_data = ir6[5];
              ir6[5] = (v1842_data + v1841_data);
              float v1844_data = r5[6];
              float v1845_data = ir6[6];
              ir6[6] = (v1845_data + v1844_data);
              float v1847_data = r5[7];
              float v1848_data = ir6[7];
              ir6[7] = (v1848_data + v1847_data);
              float v1850_data = r5[8];
              float v1851_data = ir6[8];
              ir6[8] = (v1851_data + v1850_data);
              float v1853_data = r5[9];
              float v1854_data = ir6[9];
              ir6[9] = (v1854_data + v1853_data);
              float v1856_data = r5[10];
              float v1857_data = ir6[10];
              ir6[10] = (v1857_data + v1856_data);
              float v1859_data = r5[11];
              float v1860_data = ir6[11];
              ir6[11] = (v1860_data + v1859_data);
              // r6 = ir6
              if (v25_g) {
                #pragma unroll
                for (int32_t v1862_n1 = 0; v1862_n1 < 12; ++v1862_n1) {
                  float v1864_data = ir6[v1862_n1];
                  r6[v1862_n1] = v1864_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r6);
              sycl::group_barrier(item.get_sub_group());
              if (v25_g) {
                int32_t v1870_off = v24_lead + 6;
                #pragma unroll
                for (int32_t v1865_i1 = 0; v1865_i1 < 12; ++v1865_i1) {
                  float v1867_data = r6[v1865_i1];
                  int32_t v1872_a = v1870_off + (v1865_i1 * 12);
                  s0[(v1872_a ^ ((v1872_a >> 4) & 15))] = v1867_data;
                }
              }
              float r7[12]{};
              // ir7 = +(s0)
              // [(0, 12), (0, 12)] []
              float ir7[12]{};
              int32_t v1883_sw = v24_lead ^ ((v24_lead >> 4) & 15);
              sycl::group_barrier(item.get_sub_group());
              float v1884_data_pre = s0[v34_g ? (v1883_sw) : (0)];
              float v1884_data = v34_g ? (v1884_data_pre) : (0.0f);
              float v1885_data = ir7[0];
              ir7[0] = (v1885_data + v1884_data);
              int32_t v1887_a = v24_lead + 12;
              float v1891_data_pre = s0[v34_g ? ((v1887_a ^ ((v1887_a >> 4) & 15))) : (0)];
              float v1891_data = v34_g ? (v1891_data_pre) : (0.0f);
              float v1892_data = ir7[1];
              ir7[1] = (v1892_data + v1891_data);
              int32_t v1894_a = v24_lead + 24;
              float v1898_data_pre = s0[v34_g ? ((v1894_a ^ ((v1894_a >> 4) & 15))) : (0)];
              float v1898_data = v34_g ? (v1898_data_pre) : (0.0f);
              float v1899_data = ir7[2];
              ir7[2] = (v1899_data + v1898_data);
              int32_t v1901_a = v24_lead + 36;
              float v1905_data_pre = s0[v34_g ? ((v1901_a ^ ((v1901_a >> 4) & 15))) : (0)];
              float v1905_data = v34_g ? (v1905_data_pre) : (0.0f);
              float v1906_data = ir7[3];
              ir7[3] = (v1906_data + v1905_data);
              int32_t v1908_a = v24_lead + 48;
              float v1912_data_pre = s0[v34_g ? ((v1908_a ^ ((v1908_a >> 4) & 15))) : (0)];
              float v1912_data = v34_g ? (v1912_data_pre) : (0.0f);
              float v1913_data = ir7[4];
              ir7[4] = (v1913_data + v1912_data);
              int32_t v1915_a = v24_lead + 60;
              float v1919_data_pre = s0[v34_g ? ((v1915_a ^ ((v1915_a >> 4) & 15))) : (0)];
              float v1919_data = v34_g ? (v1919_data_pre) : (0.0f);
              float v1920_data = ir7[5];
              ir7[5] = (v1920_data + v1919_data);
              int32_t v1922_a = v24_lead + 72;
              float v1926_data_pre = s0[v34_g ? ((v1922_a ^ ((v1922_a >> 4) & 15))) : (0)];
              float v1926_data = v34_g ? (v1926_data_pre) : (0.0f);
              float v1927_data = ir7[6];
              ir7[6] = (v1927_data + v1926_data);
              int32_t v1929_a = v24_lead + 84;
              float v1933_data_pre = s0[v34_g ? ((v1929_a ^ ((v1929_a >> 4) & 15))) : (0)];
              float v1933_data = v34_g ? (v1933_data_pre) : (0.0f);
              float v1934_data = ir7[7];
              ir7[7] = (v1934_data + v1933_data);
              int32_t v1936_a = v24_lead + 96;
              float v1940_data_pre = s0[v34_g ? ((v1936_a ^ ((v1936_a >> 4) & 15))) : (0)];
              float v1940_data = v34_g ? (v1940_data_pre) : (0.0f);
              float v1941_data = ir7[8];
              ir7[8] = (v1941_data + v1940_data);
              int32_t v1943_a = v24_lead + 108;
              float v1947_data_pre = s0[v34_g ? ((v1943_a ^ ((v1943_a >> 4) & 15))) : (0)];
              float v1947_data = v34_g ? (v1947_data_pre) : (0.0f);
              float v1948_data = ir7[9];
              ir7[9] = (v1948_data + v1947_data);
              int32_t v1950_a = v24_lead + 120;
              float v1954_data_pre = s0[v34_g ? ((v1950_a ^ ((v1950_a >> 4) & 15))) : (0)];
              float v1954_data = v34_g ? (v1954_data_pre) : (0.0f);
              float v1955_data = ir7[10];
              ir7[10] = (v1955_data + v1954_data);
              int32_t v1957_a = v24_lead + 132;
              float v1961_data_pre = s0[v34_g ? ((v1957_a ^ ((v1957_a >> 4) & 15))) : (0)];
              float v1961_data = v34_g ? (v1961_data_pre) : (0.0f);
              float v1962_data = ir7[11];
              ir7[11] = (v1962_data + v1961_data);
              // r7 = ir7
              if (v34_g) {
                #pragma unroll
                for (int32_t v1964_n1 = 0; v1964_n1 < 12; ++v1964_n1) {
                  float v1966_data = ir7[v1964_n1];
                  r7[v1964_n1] = v1966_data;
                }
              }
              // glb_m4 = store{r>g}(r7);
              if (v34_g) {
                #pragma unroll
                for (int32_t v1967_i1 = 0; v1967_i1 < 12; ++v1967_i1) {
                  float v1969_data = r7[v1967_i1];
                  glb_m4[(v24_lead + (v1967_i1 * 12))] = v1969_data;
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

