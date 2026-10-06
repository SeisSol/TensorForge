// === base name ===
kernel_4525b3dea5621290

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4525b3dea5621290 = {{16, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4525b3dea5621290(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4525b3dea5621290(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4525b3dea5621290(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_4525b3dea5621290(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4525b3dea5621290(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4525b3dea5621290(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4525b3dea5621290(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0) {
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
        //   m3 12×12(12×12) {0..12}×{0..12} strided
        //   m4 2×12(2×12) {0..2}×{0..12} strided
        //   m5 12×12(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
        //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
        //   m3[i,j] = t0[i,j]
        //   t0[i,j]@{6..12}×{0..12} = m4[i,k] × m1[k,j]
        //   m5[i,j] = t0[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"X","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N2","bbox":[[0,0],[2,12]],"name":"m4","ordered":false,"parts":1,"shape":[2,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[2,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[160 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[144];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v10_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v10_batchId0 < numElements0; v10_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v11_ahead1 = v10_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v10_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v10_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v10_batchId0 * 72 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v10_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v10_batchId0 * 24 + 0 + m4_extraOffset];
              float *const __restrict__ glb_m5 = &m5[v10_batchId0 * 144 + 0 + m5_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v27_lead = item.get_local_id(2) % 16;
              bool v28_g = v27_lead < 6;
              if (v28_g) {
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 12; ++v29_i1) {
                  float v34_data = glb_m0[(v27_lead + (v29_i1 * 6))];
                  r0[v29_i1] = v34_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              bool v37_g = v27_lead < 12;
              if (v37_g) {
                #pragma unroll
                for (int32_t v38_i1 = 0; v38_i1 < 12; ++v38_i1) {
                  float v43_data = glb_m1[(v27_lead + (v38_i1 * 12))];
                  r1[v38_i1] = v43_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m2);
              if (v28_g) {
                #pragma unroll
                for (int32_t v46_i1 = 0; v46_i1 < 12; ++v46_i1) {
                  float v51_data = glb_m2[(v27_lead + (v46_i1 * 6))];
                  r3[v46_i1] = v51_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[12]{};
              // r2 = +(r0 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              float v54_data = r0[0];
              float v55_data = r1[0];
              float v56_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v58_data = r2[0];
              r2[0] = (v58_data + (v54_data * v56_bc));
              float v61_data = r1[1];
              float v62_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v64_data = r2[1];
              r2[1] = (v64_data + (v54_data * v62_bc));
              float v67_data = r1[2];
              float v68_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v70_data = r2[2];
              r2[2] = (v70_data + (v54_data * v68_bc));
              float v73_data = r1[3];
              float v74_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v76_data = r2[3];
              r2[3] = (v76_data + (v54_data * v74_bc));
              float v79_data = r1[4];
              float v80_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v82_data = r2[4];
              r2[4] = (v82_data + (v54_data * v80_bc));
              float v85_data = r1[5];
              float v86_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v88_data = r2[5];
              r2[5] = (v88_data + (v54_data * v86_bc));
              float v91_data = r1[6];
              float v92_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v94_data = r2[6];
              r2[6] = (v94_data + (v54_data * v92_bc));
              float v97_data = r1[7];
              float v98_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v100_data = r2[7];
              r2[7] = (v100_data + (v54_data * v98_bc));
              float v103_data = r1[8];
              float v104_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v106_data = r2[8];
              r2[8] = (v106_data + (v54_data * v104_bc));
              float v109_data = r1[9];
              float v110_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v112_data = r2[9];
              r2[9] = (v112_data + (v54_data * v110_bc));
              float v115_data = r1[10];
              float v116_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v118_data = r2[10];
              r2[10] = (v118_data + (v54_data * v116_bc));
              float v121_data = r1[11];
              float v122_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v124_data = r2[11];
              r2[11] = (v124_data + (v54_data * v122_bc));
              float v126_data = r0[1];
              float v128_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v130_data = r2[0];
              r2[0] = (v130_data + (v126_data * v128_bc));
              float v134_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v136_data = r2[1];
              r2[1] = (v136_data + (v126_data * v134_bc));
              float v140_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v142_data = r2[2];
              r2[2] = (v142_data + (v126_data * v140_bc));
              float v146_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v148_data = r2[3];
              r2[3] = (v148_data + (v126_data * v146_bc));
              float v152_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v154_data = r2[4];
              r2[4] = (v154_data + (v126_data * v152_bc));
              float v158_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v160_data = r2[5];
              r2[5] = (v160_data + (v126_data * v158_bc));
              float v164_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v166_data = r2[6];
              r2[6] = (v166_data + (v126_data * v164_bc));
              float v170_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v172_data = r2[7];
              r2[7] = (v172_data + (v126_data * v170_bc));
              float v176_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v178_data = r2[8];
              r2[8] = (v178_data + (v126_data * v176_bc));
              float v182_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v184_data = r2[9];
              r2[9] = (v184_data + (v126_data * v182_bc));
              float v188_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v190_data = r2[10];
              r2[10] = (v190_data + (v126_data * v188_bc));
              float v194_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v196_data = r2[11];
              r2[11] = (v196_data + (v126_data * v194_bc));
              float v198_data = r0[2];
              float v200_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v202_data = r2[0];
              r2[0] = (v202_data + (v198_data * v200_bc));
              float v206_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v208_data = r2[1];
              r2[1] = (v208_data + (v198_data * v206_bc));
              float v212_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v214_data = r2[2];
              r2[2] = (v214_data + (v198_data * v212_bc));
              float v218_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v220_data = r2[3];
              r2[3] = (v220_data + (v198_data * v218_bc));
              float v224_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v226_data = r2[4];
              r2[4] = (v226_data + (v198_data * v224_bc));
              float v230_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v232_data = r2[5];
              r2[5] = (v232_data + (v198_data * v230_bc));
              float v236_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v238_data = r2[6];
              r2[6] = (v238_data + (v198_data * v236_bc));
              float v242_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v244_data = r2[7];
              r2[7] = (v244_data + (v198_data * v242_bc));
              float v248_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v250_data = r2[8];
              r2[8] = (v250_data + (v198_data * v248_bc));
              float v254_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v256_data = r2[9];
              r2[9] = (v256_data + (v198_data * v254_bc));
              float v260_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v262_data = r2[10];
              r2[10] = (v262_data + (v198_data * v260_bc));
              float v266_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v268_data = r2[11];
              r2[11] = (v268_data + (v198_data * v266_bc));
              float v270_data = r0[3];
              float v272_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v274_data = r2[0];
              r2[0] = (v274_data + (v270_data * v272_bc));
              float v278_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v280_data = r2[1];
              r2[1] = (v280_data + (v270_data * v278_bc));
              float v284_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v286_data = r2[2];
              r2[2] = (v286_data + (v270_data * v284_bc));
              float v290_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v292_data = r2[3];
              r2[3] = (v292_data + (v270_data * v290_bc));
              float v296_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v298_data = r2[4];
              r2[4] = (v298_data + (v270_data * v296_bc));
              float v302_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v304_data = r2[5];
              r2[5] = (v304_data + (v270_data * v302_bc));
              float v308_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v310_data = r2[6];
              r2[6] = (v310_data + (v270_data * v308_bc));
              float v314_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v316_data = r2[7];
              r2[7] = (v316_data + (v270_data * v314_bc));
              float v320_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v322_data = r2[8];
              r2[8] = (v322_data + (v270_data * v320_bc));
              float v326_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v328_data = r2[9];
              r2[9] = (v328_data + (v270_data * v326_bc));
              float v332_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v334_data = r2[10];
              r2[10] = (v334_data + (v270_data * v332_bc));
              float v338_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v340_data = r2[11];
              r2[11] = (v340_data + (v270_data * v338_bc));
              float v342_data = r0[4];
              float v344_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v346_data = r2[0];
              r2[0] = (v346_data + (v342_data * v344_bc));
              float v350_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v352_data = r2[1];
              r2[1] = (v352_data + (v342_data * v350_bc));
              float v356_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v358_data = r2[2];
              r2[2] = (v358_data + (v342_data * v356_bc));
              float v362_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v364_data = r2[3];
              r2[3] = (v364_data + (v342_data * v362_bc));
              float v368_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v370_data = r2[4];
              r2[4] = (v370_data + (v342_data * v368_bc));
              float v374_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v376_data = r2[5];
              r2[5] = (v376_data + (v342_data * v374_bc));
              float v380_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v382_data = r2[6];
              r2[6] = (v382_data + (v342_data * v380_bc));
              float v386_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v388_data = r2[7];
              r2[7] = (v388_data + (v342_data * v386_bc));
              float v392_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v394_data = r2[8];
              r2[8] = (v394_data + (v342_data * v392_bc));
              float v398_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v400_data = r2[9];
              r2[9] = (v400_data + (v342_data * v398_bc));
              float v404_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v406_data = r2[10];
              r2[10] = (v406_data + (v342_data * v404_bc));
              float v410_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v412_data = r2[11];
              r2[11] = (v412_data + (v342_data * v410_bc));
              float v414_data = r0[5];
              float v416_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v418_data = r2[0];
              r2[0] = (v418_data + (v414_data * v416_bc));
              float v422_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v424_data = r2[1];
              r2[1] = (v424_data + (v414_data * v422_bc));
              float v428_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v430_data = r2[2];
              r2[2] = (v430_data + (v414_data * v428_bc));
              float v434_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v436_data = r2[3];
              r2[3] = (v436_data + (v414_data * v434_bc));
              float v440_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v442_data = r2[4];
              r2[4] = (v442_data + (v414_data * v440_bc));
              float v446_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v448_data = r2[5];
              r2[5] = (v448_data + (v414_data * v446_bc));
              float v452_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v454_data = r2[6];
              r2[6] = (v454_data + (v414_data * v452_bc));
              float v458_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v460_data = r2[7];
              r2[7] = (v460_data + (v414_data * v458_bc));
              float v464_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v466_data = r2[8];
              r2[8] = (v466_data + (v414_data * v464_bc));
              float v470_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v472_data = r2[9];
              r2[9] = (v472_data + (v414_data * v470_bc));
              float v476_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v478_data = r2[10];
              r2[10] = (v478_data + (v414_data * v476_bc));
              float v482_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v484_data = r2[11];
              r2[11] = (v484_data + (v414_data * v482_bc));
              float v486_data = r0[6];
              float v488_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v490_data = r2[0];
              r2[0] = (v490_data + (v486_data * v488_bc));
              float v494_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v496_data = r2[1];
              r2[1] = (v496_data + (v486_data * v494_bc));
              float v500_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v502_data = r2[2];
              r2[2] = (v502_data + (v486_data * v500_bc));
              float v506_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v508_data = r2[3];
              r2[3] = (v508_data + (v486_data * v506_bc));
              float v512_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v514_data = r2[4];
              r2[4] = (v514_data + (v486_data * v512_bc));
              float v518_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v520_data = r2[5];
              r2[5] = (v520_data + (v486_data * v518_bc));
              float v524_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v526_data = r2[6];
              r2[6] = (v526_data + (v486_data * v524_bc));
              float v530_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v532_data = r2[7];
              r2[7] = (v532_data + (v486_data * v530_bc));
              float v536_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v538_data = r2[8];
              r2[8] = (v538_data + (v486_data * v536_bc));
              float v542_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v544_data = r2[9];
              r2[9] = (v544_data + (v486_data * v542_bc));
              float v548_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v550_data = r2[10];
              r2[10] = (v550_data + (v486_data * v548_bc));
              float v554_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v556_data = r2[11];
              r2[11] = (v556_data + (v486_data * v554_bc));
              float v558_data = r0[7];
              float v560_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v562_data = r2[0];
              r2[0] = (v562_data + (v558_data * v560_bc));
              float v566_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v568_data = r2[1];
              r2[1] = (v568_data + (v558_data * v566_bc));
              float v572_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v574_data = r2[2];
              r2[2] = (v574_data + (v558_data * v572_bc));
              float v578_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v580_data = r2[3];
              r2[3] = (v580_data + (v558_data * v578_bc));
              float v584_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v586_data = r2[4];
              r2[4] = (v586_data + (v558_data * v584_bc));
              float v590_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v592_data = r2[5];
              r2[5] = (v592_data + (v558_data * v590_bc));
              float v596_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v598_data = r2[6];
              r2[6] = (v598_data + (v558_data * v596_bc));
              float v602_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v604_data = r2[7];
              r2[7] = (v604_data + (v558_data * v602_bc));
              float v608_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v610_data = r2[8];
              r2[8] = (v610_data + (v558_data * v608_bc));
              float v614_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v616_data = r2[9];
              r2[9] = (v616_data + (v558_data * v614_bc));
              float v620_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v622_data = r2[10];
              r2[10] = (v622_data + (v558_data * v620_bc));
              float v626_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v628_data = r2[11];
              r2[11] = (v628_data + (v558_data * v626_bc));
              float v630_data = r0[8];
              float v632_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v634_data = r2[0];
              r2[0] = (v634_data + (v630_data * v632_bc));
              float v638_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v640_data = r2[1];
              r2[1] = (v640_data + (v630_data * v638_bc));
              float v644_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v646_data = r2[2];
              r2[2] = (v646_data + (v630_data * v644_bc));
              float v650_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v652_data = r2[3];
              r2[3] = (v652_data + (v630_data * v650_bc));
              float v656_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v658_data = r2[4];
              r2[4] = (v658_data + (v630_data * v656_bc));
              float v662_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v664_data = r2[5];
              r2[5] = (v664_data + (v630_data * v662_bc));
              float v668_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v670_data = r2[6];
              r2[6] = (v670_data + (v630_data * v668_bc));
              float v674_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v676_data = r2[7];
              r2[7] = (v676_data + (v630_data * v674_bc));
              float v680_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v682_data = r2[8];
              r2[8] = (v682_data + (v630_data * v680_bc));
              float v686_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v688_data = r2[9];
              r2[9] = (v688_data + (v630_data * v686_bc));
              float v692_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v694_data = r2[10];
              r2[10] = (v694_data + (v630_data * v692_bc));
              float v698_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v700_data = r2[11];
              r2[11] = (v700_data + (v630_data * v698_bc));
              float v702_data = r0[9];
              float v704_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v706_data = r2[0];
              r2[0] = (v706_data + (v702_data * v704_bc));
              float v710_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v712_data = r2[1];
              r2[1] = (v712_data + (v702_data * v710_bc));
              float v716_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v718_data = r2[2];
              r2[2] = (v718_data + (v702_data * v716_bc));
              float v722_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v724_data = r2[3];
              r2[3] = (v724_data + (v702_data * v722_bc));
              float v728_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v730_data = r2[4];
              r2[4] = (v730_data + (v702_data * v728_bc));
              float v734_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v736_data = r2[5];
              r2[5] = (v736_data + (v702_data * v734_bc));
              float v740_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v742_data = r2[6];
              r2[6] = (v742_data + (v702_data * v740_bc));
              float v746_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v748_data = r2[7];
              r2[7] = (v748_data + (v702_data * v746_bc));
              float v752_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v754_data = r2[8];
              r2[8] = (v754_data + (v702_data * v752_bc));
              float v758_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v760_data = r2[9];
              r2[9] = (v760_data + (v702_data * v758_bc));
              float v764_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v766_data = r2[10];
              r2[10] = (v766_data + (v702_data * v764_bc));
              float v770_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v772_data = r2[11];
              r2[11] = (v772_data + (v702_data * v770_bc));
              float v774_data = r0[10];
              float v776_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v778_data = r2[0];
              r2[0] = (v778_data + (v774_data * v776_bc));
              float v782_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v784_data = r2[1];
              r2[1] = (v784_data + (v774_data * v782_bc));
              float v788_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v790_data = r2[2];
              r2[2] = (v790_data + (v774_data * v788_bc));
              float v794_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v796_data = r2[3];
              r2[3] = (v796_data + (v774_data * v794_bc));
              float v800_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v802_data = r2[4];
              r2[4] = (v802_data + (v774_data * v800_bc));
              float v806_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v808_data = r2[5];
              r2[5] = (v808_data + (v774_data * v806_bc));
              float v812_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v814_data = r2[6];
              r2[6] = (v814_data + (v774_data * v812_bc));
              float v818_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v820_data = r2[7];
              r2[7] = (v820_data + (v774_data * v818_bc));
              float v824_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v826_data = r2[8];
              r2[8] = (v826_data + (v774_data * v824_bc));
              float v830_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v832_data = r2[9];
              r2[9] = (v832_data + (v774_data * v830_bc));
              float v836_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v838_data = r2[10];
              r2[10] = (v838_data + (v774_data * v836_bc));
              float v842_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v844_data = r2[11];
              r2[11] = (v844_data + (v774_data * v842_bc));
              float v846_data = r0[11];
              float v848_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v850_data = r2[0];
              r2[0] = (v850_data + (v846_data * v848_bc));
              float v854_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v856_data = r2[1];
              r2[1] = (v856_data + (v846_data * v854_bc));
              float v860_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v862_data = r2[2];
              r2[2] = (v862_data + (v846_data * v860_bc));
              float v866_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v868_data = r2[3];
              r2[3] = (v868_data + (v846_data * v866_bc));
              float v872_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v874_data = r2[4];
              r2[4] = (v874_data + (v846_data * v872_bc));
              float v878_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v880_data = r2[5];
              r2[5] = (v880_data + (v846_data * v878_bc));
              float v884_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v886_data = r2[6];
              r2[6] = (v886_data + (v846_data * v884_bc));
              float v890_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v892_data = r2[7];
              r2[7] = (v892_data + (v846_data * v890_bc));
              float v896_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v898_data = r2[8];
              r2[8] = (v898_data + (v846_data * v896_bc));
              float v902_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v904_data = r2[9];
              r2[9] = (v904_data + (v846_data * v902_bc));
              float v908_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v910_data = r2[10];
              r2[10] = (v910_data + (v846_data * v908_bc));
              float v914_bc = sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v916_data = r2[11];
              r2[11] = (v916_data + (v846_data * v914_bc));
              // s0 = store{r>s}(localShrMem0, r2);
              if (v28_g) {
                #pragma unroll
                for (int32_t v918_i1 = 0; v918_i1 < 12; ++v918_i1) {
                  float v920_data = r2[v918_i1];
                  int32_t v924_a = v27_lead + (v918_i1 * 12);
                  s0[(v924_a ^ ((v924_a >> 4) & 15))] = v920_data;
                }
              }
              float r6[12]{};
              // r6 = load{g>r}(glb_m4);
              bool v929_g = v27_lead < 2;
              if (v929_g) {
                #pragma unroll
                for (int32_t v930_i1 = 0; v930_i1 < 12; ++v930_i1) {
                  float v935_data = glb_m4[(v27_lead + (v930_i1 * 2))];
                  r6[v930_i1] = v935_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[12]{};
              // ir4 = +(r3 * r1)
              // [(0, 6), (0, 12)] [(0, 12)]
              float ir4[12]{};
              float v939_data = r3[0];
              float v943_data = ir4[0];
              ir4[0] = (v943_data + (v939_data * v56_bc));
              float v949_data = ir4[1];
              ir4[1] = (v949_data + (v939_data * v62_bc));
              float v955_data = ir4[2];
              ir4[2] = (v955_data + (v939_data * v68_bc));
              float v961_data = ir4[3];
              ir4[3] = (v961_data + (v939_data * v74_bc));
              float v967_data = ir4[4];
              ir4[4] = (v967_data + (v939_data * v80_bc));
              float v973_data = ir4[5];
              ir4[5] = (v973_data + (v939_data * v86_bc));
              float v979_data = ir4[6];
              ir4[6] = (v979_data + (v939_data * v92_bc));
              float v985_data = ir4[7];
              ir4[7] = (v985_data + (v939_data * v98_bc));
              float v991_data = ir4[8];
              ir4[8] = (v991_data + (v939_data * v104_bc));
              float v997_data = ir4[9];
              ir4[9] = (v997_data + (v939_data * v110_bc));
              float v1003_data = ir4[10];
              ir4[10] = (v1003_data + (v939_data * v116_bc));
              float v1009_data = ir4[11];
              ir4[11] = (v1009_data + (v939_data * v122_bc));
              float v1011_data = r3[1];
              float v1015_data = ir4[0];
              ir4[0] = (v1015_data + (v1011_data * v128_bc));
              float v1021_data = ir4[1];
              ir4[1] = (v1021_data + (v1011_data * v134_bc));
              float v1027_data = ir4[2];
              ir4[2] = (v1027_data + (v1011_data * v140_bc));
              float v1033_data = ir4[3];
              ir4[3] = (v1033_data + (v1011_data * v146_bc));
              float v1039_data = ir4[4];
              ir4[4] = (v1039_data + (v1011_data * v152_bc));
              float v1045_data = ir4[5];
              ir4[5] = (v1045_data + (v1011_data * v158_bc));
              float v1051_data = ir4[6];
              ir4[6] = (v1051_data + (v1011_data * v164_bc));
              float v1057_data = ir4[7];
              ir4[7] = (v1057_data + (v1011_data * v170_bc));
              float v1063_data = ir4[8];
              ir4[8] = (v1063_data + (v1011_data * v176_bc));
              float v1069_data = ir4[9];
              ir4[9] = (v1069_data + (v1011_data * v182_bc));
              float v1075_data = ir4[10];
              ir4[10] = (v1075_data + (v1011_data * v188_bc));
              float v1081_data = ir4[11];
              ir4[11] = (v1081_data + (v1011_data * v194_bc));
              float v1083_data = r3[2];
              float v1087_data = ir4[0];
              ir4[0] = (v1087_data + (v1083_data * v200_bc));
              float v1093_data = ir4[1];
              ir4[1] = (v1093_data + (v1083_data * v206_bc));
              float v1099_data = ir4[2];
              ir4[2] = (v1099_data + (v1083_data * v212_bc));
              float v1105_data = ir4[3];
              ir4[3] = (v1105_data + (v1083_data * v218_bc));
              float v1111_data = ir4[4];
              ir4[4] = (v1111_data + (v1083_data * v224_bc));
              float v1117_data = ir4[5];
              ir4[5] = (v1117_data + (v1083_data * v230_bc));
              float v1123_data = ir4[6];
              ir4[6] = (v1123_data + (v1083_data * v236_bc));
              float v1129_data = ir4[7];
              ir4[7] = (v1129_data + (v1083_data * v242_bc));
              float v1135_data = ir4[8];
              ir4[8] = (v1135_data + (v1083_data * v248_bc));
              float v1141_data = ir4[9];
              ir4[9] = (v1141_data + (v1083_data * v254_bc));
              float v1147_data = ir4[10];
              ir4[10] = (v1147_data + (v1083_data * v260_bc));
              float v1153_data = ir4[11];
              ir4[11] = (v1153_data + (v1083_data * v266_bc));
              float v1155_data = r3[3];
              float v1159_data = ir4[0];
              ir4[0] = (v1159_data + (v1155_data * v272_bc));
              float v1165_data = ir4[1];
              ir4[1] = (v1165_data + (v1155_data * v278_bc));
              float v1171_data = ir4[2];
              ir4[2] = (v1171_data + (v1155_data * v284_bc));
              float v1177_data = ir4[3];
              ir4[3] = (v1177_data + (v1155_data * v290_bc));
              float v1183_data = ir4[4];
              ir4[4] = (v1183_data + (v1155_data * v296_bc));
              float v1189_data = ir4[5];
              ir4[5] = (v1189_data + (v1155_data * v302_bc));
              float v1195_data = ir4[6];
              ir4[6] = (v1195_data + (v1155_data * v308_bc));
              float v1201_data = ir4[7];
              ir4[7] = (v1201_data + (v1155_data * v314_bc));
              float v1207_data = ir4[8];
              ir4[8] = (v1207_data + (v1155_data * v320_bc));
              float v1213_data = ir4[9];
              ir4[9] = (v1213_data + (v1155_data * v326_bc));
              float v1219_data = ir4[10];
              ir4[10] = (v1219_data + (v1155_data * v332_bc));
              float v1225_data = ir4[11];
              ir4[11] = (v1225_data + (v1155_data * v338_bc));
              float v1227_data = r3[4];
              float v1231_data = ir4[0];
              ir4[0] = (v1231_data + (v1227_data * v344_bc));
              float v1237_data = ir4[1];
              ir4[1] = (v1237_data + (v1227_data * v350_bc));
              float v1243_data = ir4[2];
              ir4[2] = (v1243_data + (v1227_data * v356_bc));
              float v1249_data = ir4[3];
              ir4[3] = (v1249_data + (v1227_data * v362_bc));
              float v1255_data = ir4[4];
              ir4[4] = (v1255_data + (v1227_data * v368_bc));
              float v1261_data = ir4[5];
              ir4[5] = (v1261_data + (v1227_data * v374_bc));
              float v1267_data = ir4[6];
              ir4[6] = (v1267_data + (v1227_data * v380_bc));
              float v1273_data = ir4[7];
              ir4[7] = (v1273_data + (v1227_data * v386_bc));
              float v1279_data = ir4[8];
              ir4[8] = (v1279_data + (v1227_data * v392_bc));
              float v1285_data = ir4[9];
              ir4[9] = (v1285_data + (v1227_data * v398_bc));
              float v1291_data = ir4[10];
              ir4[10] = (v1291_data + (v1227_data * v404_bc));
              float v1297_data = ir4[11];
              ir4[11] = (v1297_data + (v1227_data * v410_bc));
              float v1299_data = r3[5];
              float v1303_data = ir4[0];
              ir4[0] = (v1303_data + (v1299_data * v416_bc));
              float v1309_data = ir4[1];
              ir4[1] = (v1309_data + (v1299_data * v422_bc));
              float v1315_data = ir4[2];
              ir4[2] = (v1315_data + (v1299_data * v428_bc));
              float v1321_data = ir4[3];
              ir4[3] = (v1321_data + (v1299_data * v434_bc));
              float v1327_data = ir4[4];
              ir4[4] = (v1327_data + (v1299_data * v440_bc));
              float v1333_data = ir4[5];
              ir4[5] = (v1333_data + (v1299_data * v446_bc));
              float v1339_data = ir4[6];
              ir4[6] = (v1339_data + (v1299_data * v452_bc));
              float v1345_data = ir4[7];
              ir4[7] = (v1345_data + (v1299_data * v458_bc));
              float v1351_data = ir4[8];
              ir4[8] = (v1351_data + (v1299_data * v464_bc));
              float v1357_data = ir4[9];
              ir4[9] = (v1357_data + (v1299_data * v470_bc));
              float v1363_data = ir4[10];
              ir4[10] = (v1363_data + (v1299_data * v476_bc));
              float v1369_data = ir4[11];
              ir4[11] = (v1369_data + (v1299_data * v482_bc));
              float v1371_data = r3[6];
              float v1375_data = ir4[0];
              ir4[0] = (v1375_data + (v1371_data * v488_bc));
              float v1381_data = ir4[1];
              ir4[1] = (v1381_data + (v1371_data * v494_bc));
              float v1387_data = ir4[2];
              ir4[2] = (v1387_data + (v1371_data * v500_bc));
              float v1393_data = ir4[3];
              ir4[3] = (v1393_data + (v1371_data * v506_bc));
              float v1399_data = ir4[4];
              ir4[4] = (v1399_data + (v1371_data * v512_bc));
              float v1405_data = ir4[5];
              ir4[5] = (v1405_data + (v1371_data * v518_bc));
              float v1411_data = ir4[6];
              ir4[6] = (v1411_data + (v1371_data * v524_bc));
              float v1417_data = ir4[7];
              ir4[7] = (v1417_data + (v1371_data * v530_bc));
              float v1423_data = ir4[8];
              ir4[8] = (v1423_data + (v1371_data * v536_bc));
              float v1429_data = ir4[9];
              ir4[9] = (v1429_data + (v1371_data * v542_bc));
              float v1435_data = ir4[10];
              ir4[10] = (v1435_data + (v1371_data * v548_bc));
              float v1441_data = ir4[11];
              ir4[11] = (v1441_data + (v1371_data * v554_bc));
              float v1443_data = r3[7];
              float v1447_data = ir4[0];
              ir4[0] = (v1447_data + (v1443_data * v560_bc));
              float v1453_data = ir4[1];
              ir4[1] = (v1453_data + (v1443_data * v566_bc));
              float v1459_data = ir4[2];
              ir4[2] = (v1459_data + (v1443_data * v572_bc));
              float v1465_data = ir4[3];
              ir4[3] = (v1465_data + (v1443_data * v578_bc));
              float v1471_data = ir4[4];
              ir4[4] = (v1471_data + (v1443_data * v584_bc));
              float v1477_data = ir4[5];
              ir4[5] = (v1477_data + (v1443_data * v590_bc));
              float v1483_data = ir4[6];
              ir4[6] = (v1483_data + (v1443_data * v596_bc));
              float v1489_data = ir4[7];
              ir4[7] = (v1489_data + (v1443_data * v602_bc));
              float v1495_data = ir4[8];
              ir4[8] = (v1495_data + (v1443_data * v608_bc));
              float v1501_data = ir4[9];
              ir4[9] = (v1501_data + (v1443_data * v614_bc));
              float v1507_data = ir4[10];
              ir4[10] = (v1507_data + (v1443_data * v620_bc));
              float v1513_data = ir4[11];
              ir4[11] = (v1513_data + (v1443_data * v626_bc));
              float v1515_data = r3[8];
              float v1519_data = ir4[0];
              ir4[0] = (v1519_data + (v1515_data * v632_bc));
              float v1525_data = ir4[1];
              ir4[1] = (v1525_data + (v1515_data * v638_bc));
              float v1531_data = ir4[2];
              ir4[2] = (v1531_data + (v1515_data * v644_bc));
              float v1537_data = ir4[3];
              ir4[3] = (v1537_data + (v1515_data * v650_bc));
              float v1543_data = ir4[4];
              ir4[4] = (v1543_data + (v1515_data * v656_bc));
              float v1549_data = ir4[5];
              ir4[5] = (v1549_data + (v1515_data * v662_bc));
              float v1555_data = ir4[6];
              ir4[6] = (v1555_data + (v1515_data * v668_bc));
              float v1561_data = ir4[7];
              ir4[7] = (v1561_data + (v1515_data * v674_bc));
              float v1567_data = ir4[8];
              ir4[8] = (v1567_data + (v1515_data * v680_bc));
              float v1573_data = ir4[9];
              ir4[9] = (v1573_data + (v1515_data * v686_bc));
              float v1579_data = ir4[10];
              ir4[10] = (v1579_data + (v1515_data * v692_bc));
              float v1585_data = ir4[11];
              ir4[11] = (v1585_data + (v1515_data * v698_bc));
              float v1587_data = r3[9];
              float v1591_data = ir4[0];
              ir4[0] = (v1591_data + (v1587_data * v704_bc));
              float v1597_data = ir4[1];
              ir4[1] = (v1597_data + (v1587_data * v710_bc));
              float v1603_data = ir4[2];
              ir4[2] = (v1603_data + (v1587_data * v716_bc));
              float v1609_data = ir4[3];
              ir4[3] = (v1609_data + (v1587_data * v722_bc));
              float v1615_data = ir4[4];
              ir4[4] = (v1615_data + (v1587_data * v728_bc));
              float v1621_data = ir4[5];
              ir4[5] = (v1621_data + (v1587_data * v734_bc));
              float v1627_data = ir4[6];
              ir4[6] = (v1627_data + (v1587_data * v740_bc));
              float v1633_data = ir4[7];
              ir4[7] = (v1633_data + (v1587_data * v746_bc));
              float v1639_data = ir4[8];
              ir4[8] = (v1639_data + (v1587_data * v752_bc));
              float v1645_data = ir4[9];
              ir4[9] = (v1645_data + (v1587_data * v758_bc));
              float v1651_data = ir4[10];
              ir4[10] = (v1651_data + (v1587_data * v764_bc));
              float v1657_data = ir4[11];
              ir4[11] = (v1657_data + (v1587_data * v770_bc));
              float v1659_data = r3[10];
              float v1663_data = ir4[0];
              ir4[0] = (v1663_data + (v1659_data * v776_bc));
              float v1669_data = ir4[1];
              ir4[1] = (v1669_data + (v1659_data * v782_bc));
              float v1675_data = ir4[2];
              ir4[2] = (v1675_data + (v1659_data * v788_bc));
              float v1681_data = ir4[3];
              ir4[3] = (v1681_data + (v1659_data * v794_bc));
              float v1687_data = ir4[4];
              ir4[4] = (v1687_data + (v1659_data * v800_bc));
              float v1693_data = ir4[5];
              ir4[5] = (v1693_data + (v1659_data * v806_bc));
              float v1699_data = ir4[6];
              ir4[6] = (v1699_data + (v1659_data * v812_bc));
              float v1705_data = ir4[7];
              ir4[7] = (v1705_data + (v1659_data * v818_bc));
              float v1711_data = ir4[8];
              ir4[8] = (v1711_data + (v1659_data * v824_bc));
              float v1717_data = ir4[9];
              ir4[9] = (v1717_data + (v1659_data * v830_bc));
              float v1723_data = ir4[10];
              ir4[10] = (v1723_data + (v1659_data * v836_bc));
              float v1729_data = ir4[11];
              ir4[11] = (v1729_data + (v1659_data * v842_bc));
              float v1731_data = r3[11];
              float v1735_data = ir4[0];
              ir4[0] = (v1735_data + (v1731_data * v848_bc));
              float v1741_data = ir4[1];
              ir4[1] = (v1741_data + (v1731_data * v854_bc));
              float v1747_data = ir4[2];
              ir4[2] = (v1747_data + (v1731_data * v860_bc));
              float v1753_data = ir4[3];
              ir4[3] = (v1753_data + (v1731_data * v866_bc));
              float v1759_data = ir4[4];
              ir4[4] = (v1759_data + (v1731_data * v872_bc));
              float v1765_data = ir4[5];
              ir4[5] = (v1765_data + (v1731_data * v878_bc));
              float v1771_data = ir4[6];
              ir4[6] = (v1771_data + (v1731_data * v884_bc));
              float v1777_data = ir4[7];
              ir4[7] = (v1777_data + (v1731_data * v890_bc));
              float v1783_data = ir4[8];
              ir4[8] = (v1783_data + (v1731_data * v896_bc));
              float v1789_data = ir4[9];
              ir4[9] = (v1789_data + (v1731_data * v902_bc));
              float v1795_data = ir4[10];
              ir4[10] = (v1795_data + (v1731_data * v908_bc));
              float v1801_data = ir4[11];
              ir4[11] = (v1801_data + (v1731_data * v914_bc));
              // r4 = ir4
              if (v28_g) {
                #pragma unroll
                for (int32_t v1803_n1 = 0; v1803_n1 < 12; ++v1803_n1) {
                  float v1805_data = ir4[v1803_n1];
                  r4[v1803_n1] = v1805_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r4);
              if (v28_g) {
                int32_t v1811_off = v27_lead + 6;
                #pragma unroll
                for (int32_t v1806_i1 = 0; v1806_i1 < 12; ++v1806_i1) {
                  float v1808_data = r4[v1806_i1];
                  int32_t v1813_a = v1811_off + (v1806_i1 * 12);
                  s0[(v1813_a ^ ((v1813_a >> 4) & 15))] = v1808_data;
                }
              }
              float r5[12]{};
              sycl::group_barrier(item.get_sub_group());
              // ir5 = +(s0)
              // [(0, 12), (0, 12)] []
              float ir5[12]{};
              int32_t v1823_sw = (v27_lead >> 4) & 15;
              float v1825_data_pre = s0[v37_g ? ((v27_lead ^ v1823_sw)) : (0)];
              float v1825_data = v37_g ? (v1825_data_pre) : (0.0f);
              float v1826_data = ir5[0];
              ir5[0] = (v1826_data + v1825_data);
              int32_t v1828_a = v27_lead + 12;
              int32_t v1829_sw = v1828_a >> 4;
              float v1832_data_pre = s0[v37_g ? ((v1828_a ^ (v1829_sw & 15))) : (0)];
              float v1832_data = v37_g ? (v1832_data_pre) : (0.0f);
              float v1833_data = ir5[1];
              ir5[1] = (v1833_data + v1832_data);
              int32_t v1835_a = v27_lead + 24;
              int32_t v1836_sw = v1835_a >> 4;
              float v1839_data_pre = s0[v37_g ? ((v1835_a ^ (v1836_sw & 15))) : (0)];
              float v1839_data = v37_g ? (v1839_data_pre) : (0.0f);
              float v1840_data = ir5[2];
              ir5[2] = (v1840_data + v1839_data);
              int32_t v1842_a = v27_lead + 36;
              int32_t v1843_sw = v1842_a >> 4;
              float v1846_data_pre = s0[v37_g ? ((v1842_a ^ (v1843_sw & 15))) : (0)];
              float v1846_data = v37_g ? (v1846_data_pre) : (0.0f);
              float v1847_data = ir5[3];
              ir5[3] = (v1847_data + v1846_data);
              int32_t v1849_a = v27_lead + 48;
              int32_t v1850_sw = v1849_a >> 4;
              float v1853_data_pre = s0[v37_g ? ((v1849_a ^ (v1850_sw & 15))) : (0)];
              float v1853_data = v37_g ? (v1853_data_pre) : (0.0f);
              float v1854_data = ir5[4];
              ir5[4] = (v1854_data + v1853_data);
              int32_t v1856_a = v27_lead + 60;
              int32_t v1857_sw = v1856_a >> 4;
              float v1860_data_pre = s0[v37_g ? ((v1856_a ^ (v1857_sw & 15))) : (0)];
              float v1860_data = v37_g ? (v1860_data_pre) : (0.0f);
              float v1861_data = ir5[5];
              ir5[5] = (v1861_data + v1860_data);
              int32_t v1863_a = v27_lead + 72;
              int32_t v1864_sw = v1863_a >> 4;
              float v1867_data_pre = s0[v37_g ? ((v1863_a ^ (v1864_sw & 15))) : (0)];
              float v1867_data = v37_g ? (v1867_data_pre) : (0.0f);
              float v1868_data = ir5[6];
              ir5[6] = (v1868_data + v1867_data);
              int32_t v1870_a = v27_lead + 84;
              int32_t v1871_sw = v1870_a >> 4;
              float v1874_data_pre = s0[v37_g ? ((v1870_a ^ (v1871_sw & 15))) : (0)];
              float v1874_data = v37_g ? (v1874_data_pre) : (0.0f);
              float v1875_data = ir5[7];
              ir5[7] = (v1875_data + v1874_data);
              int32_t v1877_a = v27_lead + 96;
              int32_t v1878_sw = v1877_a >> 4;
              float v1881_data_pre = s0[v37_g ? ((v1877_a ^ (v1878_sw & 15))) : (0)];
              float v1881_data = v37_g ? (v1881_data_pre) : (0.0f);
              float v1882_data = ir5[8];
              ir5[8] = (v1882_data + v1881_data);
              int32_t v1884_a = v27_lead + 108;
              int32_t v1885_sw = v1884_a >> 4;
              float v1888_data_pre = s0[v37_g ? ((v1884_a ^ (v1885_sw & 15))) : (0)];
              float v1888_data = v37_g ? (v1888_data_pre) : (0.0f);
              float v1889_data = ir5[9];
              ir5[9] = (v1889_data + v1888_data);
              int32_t v1891_a = v27_lead + 120;
              int32_t v1892_sw = v1891_a >> 4;
              float v1895_data_pre = s0[v37_g ? ((v1891_a ^ (v1892_sw & 15))) : (0)];
              float v1895_data = v37_g ? (v1895_data_pre) : (0.0f);
              float v1896_data = ir5[10];
              ir5[10] = (v1896_data + v1895_data);
              int32_t v1898_a = v27_lead + 132;
              int32_t v1899_sw = v1898_a >> 4;
              float v1902_data_pre = s0[v37_g ? ((v1898_a ^ (v1899_sw & 15))) : (0)];
              float v1902_data = v37_g ? (v1902_data_pre) : (0.0f);
              float v1903_data = ir5[11];
              ir5[11] = (v1903_data + v1902_data);
              // r5 = ir5
              if (v37_g) {
                #pragma unroll
                for (int32_t v1905_n1 = 0; v1905_n1 < 12; ++v1905_n1) {
                  float v1907_data = ir5[v1905_n1];
                  r5[v1905_n1] = v1907_data;
                }
              }
              // glb_m3 = store{r>g}(r5);
              if (v37_g) {
                #pragma unroll
                for (int32_t v1908_i1 = 0; v1908_i1 < 12; ++v1908_i1) {
                  float v1910_data = r5[v1908_i1];
                  glb_m3[(v27_lead + (v1908_i1 * 12))] = v1910_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m4););
              float r7[12]{};
              // ir7 = +(r6 * r1)
              // [(0, 2), (0, 12)] [(0, 12)]
              float ir7[12]{};
              float v1917_data = r6[0];
              float v1921_data = ir7[0];
              ir7[0] = (v1921_data + (v1917_data * v56_bc));
              float v1927_data = ir7[1];
              ir7[1] = (v1927_data + (v1917_data * v62_bc));
              float v1933_data = ir7[2];
              ir7[2] = (v1933_data + (v1917_data * v68_bc));
              float v1939_data = ir7[3];
              ir7[3] = (v1939_data + (v1917_data * v74_bc));
              float v1945_data = ir7[4];
              ir7[4] = (v1945_data + (v1917_data * v80_bc));
              float v1951_data = ir7[5];
              ir7[5] = (v1951_data + (v1917_data * v86_bc));
              float v1957_data = ir7[6];
              ir7[6] = (v1957_data + (v1917_data * v92_bc));
              float v1963_data = ir7[7];
              ir7[7] = (v1963_data + (v1917_data * v98_bc));
              float v1969_data = ir7[8];
              ir7[8] = (v1969_data + (v1917_data * v104_bc));
              float v1975_data = ir7[9];
              ir7[9] = (v1975_data + (v1917_data * v110_bc));
              float v1981_data = ir7[10];
              ir7[10] = (v1981_data + (v1917_data * v116_bc));
              float v1987_data = ir7[11];
              ir7[11] = (v1987_data + (v1917_data * v122_bc));
              float v1989_data = r6[1];
              float v1993_data = ir7[0];
              ir7[0] = (v1993_data + (v1989_data * v128_bc));
              float v1999_data = ir7[1];
              ir7[1] = (v1999_data + (v1989_data * v134_bc));
              float v2005_data = ir7[2];
              ir7[2] = (v2005_data + (v1989_data * v140_bc));
              float v2011_data = ir7[3];
              ir7[3] = (v2011_data + (v1989_data * v146_bc));
              float v2017_data = ir7[4];
              ir7[4] = (v2017_data + (v1989_data * v152_bc));
              float v2023_data = ir7[5];
              ir7[5] = (v2023_data + (v1989_data * v158_bc));
              float v2029_data = ir7[6];
              ir7[6] = (v2029_data + (v1989_data * v164_bc));
              float v2035_data = ir7[7];
              ir7[7] = (v2035_data + (v1989_data * v170_bc));
              float v2041_data = ir7[8];
              ir7[8] = (v2041_data + (v1989_data * v176_bc));
              float v2047_data = ir7[9];
              ir7[9] = (v2047_data + (v1989_data * v182_bc));
              float v2053_data = ir7[10];
              ir7[10] = (v2053_data + (v1989_data * v188_bc));
              float v2059_data = ir7[11];
              ir7[11] = (v2059_data + (v1989_data * v194_bc));
              float v2061_data = r6[2];
              float v2065_data = ir7[0];
              ir7[0] = (v2065_data + (v2061_data * v200_bc));
              float v2071_data = ir7[1];
              ir7[1] = (v2071_data + (v2061_data * v206_bc));
              float v2077_data = ir7[2];
              ir7[2] = (v2077_data + (v2061_data * v212_bc));
              float v2083_data = ir7[3];
              ir7[3] = (v2083_data + (v2061_data * v218_bc));
              float v2089_data = ir7[4];
              ir7[4] = (v2089_data + (v2061_data * v224_bc));
              float v2095_data = ir7[5];
              ir7[5] = (v2095_data + (v2061_data * v230_bc));
              float v2101_data = ir7[6];
              ir7[6] = (v2101_data + (v2061_data * v236_bc));
              float v2107_data = ir7[7];
              ir7[7] = (v2107_data + (v2061_data * v242_bc));
              float v2113_data = ir7[8];
              ir7[8] = (v2113_data + (v2061_data * v248_bc));
              float v2119_data = ir7[9];
              ir7[9] = (v2119_data + (v2061_data * v254_bc));
              float v2125_data = ir7[10];
              ir7[10] = (v2125_data + (v2061_data * v260_bc));
              float v2131_data = ir7[11];
              ir7[11] = (v2131_data + (v2061_data * v266_bc));
              float v2133_data = r6[3];
              float v2137_data = ir7[0];
              ir7[0] = (v2137_data + (v2133_data * v272_bc));
              float v2143_data = ir7[1];
              ir7[1] = (v2143_data + (v2133_data * v278_bc));
              float v2149_data = ir7[2];
              ir7[2] = (v2149_data + (v2133_data * v284_bc));
              float v2155_data = ir7[3];
              ir7[3] = (v2155_data + (v2133_data * v290_bc));
              float v2161_data = ir7[4];
              ir7[4] = (v2161_data + (v2133_data * v296_bc));
              float v2167_data = ir7[5];
              ir7[5] = (v2167_data + (v2133_data * v302_bc));
              float v2173_data = ir7[6];
              ir7[6] = (v2173_data + (v2133_data * v308_bc));
              float v2179_data = ir7[7];
              ir7[7] = (v2179_data + (v2133_data * v314_bc));
              float v2185_data = ir7[8];
              ir7[8] = (v2185_data + (v2133_data * v320_bc));
              float v2191_data = ir7[9];
              ir7[9] = (v2191_data + (v2133_data * v326_bc));
              float v2197_data = ir7[10];
              ir7[10] = (v2197_data + (v2133_data * v332_bc));
              float v2203_data = ir7[11];
              ir7[11] = (v2203_data + (v2133_data * v338_bc));
              float v2205_data = r6[4];
              float v2209_data = ir7[0];
              ir7[0] = (v2209_data + (v2205_data * v344_bc));
              float v2215_data = ir7[1];
              ir7[1] = (v2215_data + (v2205_data * v350_bc));
              float v2221_data = ir7[2];
              ir7[2] = (v2221_data + (v2205_data * v356_bc));
              float v2227_data = ir7[3];
              ir7[3] = (v2227_data + (v2205_data * v362_bc));
              float v2233_data = ir7[4];
              ir7[4] = (v2233_data + (v2205_data * v368_bc));
              float v2239_data = ir7[5];
              ir7[5] = (v2239_data + (v2205_data * v374_bc));
              float v2245_data = ir7[6];
              ir7[6] = (v2245_data + (v2205_data * v380_bc));
              float v2251_data = ir7[7];
              ir7[7] = (v2251_data + (v2205_data * v386_bc));
              float v2257_data = ir7[8];
              ir7[8] = (v2257_data + (v2205_data * v392_bc));
              float v2263_data = ir7[9];
              ir7[9] = (v2263_data + (v2205_data * v398_bc));
              float v2269_data = ir7[10];
              ir7[10] = (v2269_data + (v2205_data * v404_bc));
              float v2275_data = ir7[11];
              ir7[11] = (v2275_data + (v2205_data * v410_bc));
              float v2277_data = r6[5];
              float v2281_data = ir7[0];
              ir7[0] = (v2281_data + (v2277_data * v416_bc));
              float v2287_data = ir7[1];
              ir7[1] = (v2287_data + (v2277_data * v422_bc));
              float v2293_data = ir7[2];
              ir7[2] = (v2293_data + (v2277_data * v428_bc));
              float v2299_data = ir7[3];
              ir7[3] = (v2299_data + (v2277_data * v434_bc));
              float v2305_data = ir7[4];
              ir7[4] = (v2305_data + (v2277_data * v440_bc));
              float v2311_data = ir7[5];
              ir7[5] = (v2311_data + (v2277_data * v446_bc));
              float v2317_data = ir7[6];
              ir7[6] = (v2317_data + (v2277_data * v452_bc));
              float v2323_data = ir7[7];
              ir7[7] = (v2323_data + (v2277_data * v458_bc));
              float v2329_data = ir7[8];
              ir7[8] = (v2329_data + (v2277_data * v464_bc));
              float v2335_data = ir7[9];
              ir7[9] = (v2335_data + (v2277_data * v470_bc));
              float v2341_data = ir7[10];
              ir7[10] = (v2341_data + (v2277_data * v476_bc));
              float v2347_data = ir7[11];
              ir7[11] = (v2347_data + (v2277_data * v482_bc));
              float v2349_data = r6[6];
              float v2353_data = ir7[0];
              ir7[0] = (v2353_data + (v2349_data * v488_bc));
              float v2359_data = ir7[1];
              ir7[1] = (v2359_data + (v2349_data * v494_bc));
              float v2365_data = ir7[2];
              ir7[2] = (v2365_data + (v2349_data * v500_bc));
              float v2371_data = ir7[3];
              ir7[3] = (v2371_data + (v2349_data * v506_bc));
              float v2377_data = ir7[4];
              ir7[4] = (v2377_data + (v2349_data * v512_bc));
              float v2383_data = ir7[5];
              ir7[5] = (v2383_data + (v2349_data * v518_bc));
              float v2389_data = ir7[6];
              ir7[6] = (v2389_data + (v2349_data * v524_bc));
              float v2395_data = ir7[7];
              ir7[7] = (v2395_data + (v2349_data * v530_bc));
              float v2401_data = ir7[8];
              ir7[8] = (v2401_data + (v2349_data * v536_bc));
              float v2407_data = ir7[9];
              ir7[9] = (v2407_data + (v2349_data * v542_bc));
              float v2413_data = ir7[10];
              ir7[10] = (v2413_data + (v2349_data * v548_bc));
              float v2419_data = ir7[11];
              ir7[11] = (v2419_data + (v2349_data * v554_bc));
              float v2421_data = r6[7];
              float v2425_data = ir7[0];
              ir7[0] = (v2425_data + (v2421_data * v560_bc));
              float v2431_data = ir7[1];
              ir7[1] = (v2431_data + (v2421_data * v566_bc));
              float v2437_data = ir7[2];
              ir7[2] = (v2437_data + (v2421_data * v572_bc));
              float v2443_data = ir7[3];
              ir7[3] = (v2443_data + (v2421_data * v578_bc));
              float v2449_data = ir7[4];
              ir7[4] = (v2449_data + (v2421_data * v584_bc));
              float v2455_data = ir7[5];
              ir7[5] = (v2455_data + (v2421_data * v590_bc));
              float v2461_data = ir7[6];
              ir7[6] = (v2461_data + (v2421_data * v596_bc));
              float v2467_data = ir7[7];
              ir7[7] = (v2467_data + (v2421_data * v602_bc));
              float v2473_data = ir7[8];
              ir7[8] = (v2473_data + (v2421_data * v608_bc));
              float v2479_data = ir7[9];
              ir7[9] = (v2479_data + (v2421_data * v614_bc));
              float v2485_data = ir7[10];
              ir7[10] = (v2485_data + (v2421_data * v620_bc));
              float v2491_data = ir7[11];
              ir7[11] = (v2491_data + (v2421_data * v626_bc));
              float v2493_data = r6[8];
              float v2497_data = ir7[0];
              ir7[0] = (v2497_data + (v2493_data * v632_bc));
              float v2503_data = ir7[1];
              ir7[1] = (v2503_data + (v2493_data * v638_bc));
              float v2509_data = ir7[2];
              ir7[2] = (v2509_data + (v2493_data * v644_bc));
              float v2515_data = ir7[3];
              ir7[3] = (v2515_data + (v2493_data * v650_bc));
              float v2521_data = ir7[4];
              ir7[4] = (v2521_data + (v2493_data * v656_bc));
              float v2527_data = ir7[5];
              ir7[5] = (v2527_data + (v2493_data * v662_bc));
              float v2533_data = ir7[6];
              ir7[6] = (v2533_data + (v2493_data * v668_bc));
              float v2539_data = ir7[7];
              ir7[7] = (v2539_data + (v2493_data * v674_bc));
              float v2545_data = ir7[8];
              ir7[8] = (v2545_data + (v2493_data * v680_bc));
              float v2551_data = ir7[9];
              ir7[9] = (v2551_data + (v2493_data * v686_bc));
              float v2557_data = ir7[10];
              ir7[10] = (v2557_data + (v2493_data * v692_bc));
              float v2563_data = ir7[11];
              ir7[11] = (v2563_data + (v2493_data * v698_bc));
              float v2565_data = r6[9];
              float v2569_data = ir7[0];
              ir7[0] = (v2569_data + (v2565_data * v704_bc));
              float v2575_data = ir7[1];
              ir7[1] = (v2575_data + (v2565_data * v710_bc));
              float v2581_data = ir7[2];
              ir7[2] = (v2581_data + (v2565_data * v716_bc));
              float v2587_data = ir7[3];
              ir7[3] = (v2587_data + (v2565_data * v722_bc));
              float v2593_data = ir7[4];
              ir7[4] = (v2593_data + (v2565_data * v728_bc));
              float v2599_data = ir7[5];
              ir7[5] = (v2599_data + (v2565_data * v734_bc));
              float v2605_data = ir7[6];
              ir7[6] = (v2605_data + (v2565_data * v740_bc));
              float v2611_data = ir7[7];
              ir7[7] = (v2611_data + (v2565_data * v746_bc));
              float v2617_data = ir7[8];
              ir7[8] = (v2617_data + (v2565_data * v752_bc));
              float v2623_data = ir7[9];
              ir7[9] = (v2623_data + (v2565_data * v758_bc));
              float v2629_data = ir7[10];
              ir7[10] = (v2629_data + (v2565_data * v764_bc));
              float v2635_data = ir7[11];
              ir7[11] = (v2635_data + (v2565_data * v770_bc));
              float v2637_data = r6[10];
              float v2641_data = ir7[0];
              ir7[0] = (v2641_data + (v2637_data * v776_bc));
              float v2647_data = ir7[1];
              ir7[1] = (v2647_data + (v2637_data * v782_bc));
              float v2653_data = ir7[2];
              ir7[2] = (v2653_data + (v2637_data * v788_bc));
              float v2659_data = ir7[3];
              ir7[3] = (v2659_data + (v2637_data * v794_bc));
              float v2665_data = ir7[4];
              ir7[4] = (v2665_data + (v2637_data * v800_bc));
              float v2671_data = ir7[5];
              ir7[5] = (v2671_data + (v2637_data * v806_bc));
              float v2677_data = ir7[6];
              ir7[6] = (v2677_data + (v2637_data * v812_bc));
              float v2683_data = ir7[7];
              ir7[7] = (v2683_data + (v2637_data * v818_bc));
              float v2689_data = ir7[8];
              ir7[8] = (v2689_data + (v2637_data * v824_bc));
              float v2695_data = ir7[9];
              ir7[9] = (v2695_data + (v2637_data * v830_bc));
              float v2701_data = ir7[10];
              ir7[10] = (v2701_data + (v2637_data * v836_bc));
              float v2707_data = ir7[11];
              ir7[11] = (v2707_data + (v2637_data * v842_bc));
              float v2709_data = r6[11];
              float v2713_data = ir7[0];
              ir7[0] = (v2713_data + (v2709_data * v848_bc));
              float v2719_data = ir7[1];
              ir7[1] = (v2719_data + (v2709_data * v854_bc));
              float v2725_data = ir7[2];
              ir7[2] = (v2725_data + (v2709_data * v860_bc));
              float v2731_data = ir7[3];
              ir7[3] = (v2731_data + (v2709_data * v866_bc));
              float v2737_data = ir7[4];
              ir7[4] = (v2737_data + (v2709_data * v872_bc));
              float v2743_data = ir7[5];
              ir7[5] = (v2743_data + (v2709_data * v878_bc));
              float v2749_data = ir7[6];
              ir7[6] = (v2749_data + (v2709_data * v884_bc));
              float v2755_data = ir7[7];
              ir7[7] = (v2755_data + (v2709_data * v890_bc));
              float v2761_data = ir7[8];
              ir7[8] = (v2761_data + (v2709_data * v896_bc));
              float v2767_data = ir7[9];
              ir7[9] = (v2767_data + (v2709_data * v902_bc));
              float v2773_data = ir7[10];
              ir7[10] = (v2773_data + (v2709_data * v908_bc));
              float v2779_data = ir7[11];
              ir7[11] = (v2779_data + (v2709_data * v914_bc));
              // r7 = ir7
              if (v929_g) {
                #pragma unroll
                for (int32_t v2781_n1 = 0; v2781_n1 < 12; ++v2781_n1) {
                  float v2783_data = ir7[v2781_n1];
                  r7[v2781_n1] = v2783_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // s0 = store{r>s, clear}(localShrMem0, r7);
              if ((v27_lead >= 8) && v37_g) {
                #pragma unroll
                for (int32_t v2786_z1 = 0; v2786_z1 < 12; ++v2786_z1) {
                  int32_t v2791_a = v27_lead + (v2786_z1 * 12);
                  s0[(v2791_a ^ ((v2791_a >> 4) & 15))] = 0.0f;
                }
              }
              if (v929_g) {
                int32_t v2800_off = v27_lead + 6;
                #pragma unroll
                for (int32_t v2795_i1 = 0; v2795_i1 < 12; ++v2795_i1) {
                  float v2797_data = r7[v2795_i1];
                  int32_t v2802_a = v2800_off + (v2795_i1 * 12);
                  s0[(v2802_a ^ ((v2802_a >> 4) & 15))] = v2797_data;
                }
              }
              float r8[12]{};
              sycl::group_barrier(item.get_sub_group());
              // ir8 = +(s0)
              // [(0, 12), (0, 12)] []
              float ir8[12]{};
              float v2814_data_pre = s0[v37_g ? ((v27_lead ^ v1823_sw)) : (0)];
              float v2814_data = v37_g ? (v2814_data_pre) : (0.0f);
              float v2815_data = ir8[0];
              ir8[0] = (v2815_data + v2814_data);
              float v2821_data_pre = s0[v37_g ? ((v1828_a ^ (v1829_sw & 15))) : (0)];
              float v2821_data = v37_g ? (v2821_data_pre) : (0.0f);
              float v2822_data = ir8[1];
              ir8[1] = (v2822_data + v2821_data);
              float v2828_data_pre = s0[v37_g ? ((v1835_a ^ (v1836_sw & 15))) : (0)];
              float v2828_data = v37_g ? (v2828_data_pre) : (0.0f);
              float v2829_data = ir8[2];
              ir8[2] = (v2829_data + v2828_data);
              float v2835_data_pre = s0[v37_g ? ((v1842_a ^ (v1843_sw & 15))) : (0)];
              float v2835_data = v37_g ? (v2835_data_pre) : (0.0f);
              float v2836_data = ir8[3];
              ir8[3] = (v2836_data + v2835_data);
              float v2842_data_pre = s0[v37_g ? ((v1849_a ^ (v1850_sw & 15))) : (0)];
              float v2842_data = v37_g ? (v2842_data_pre) : (0.0f);
              float v2843_data = ir8[4];
              ir8[4] = (v2843_data + v2842_data);
              float v2849_data_pre = s0[v37_g ? ((v1856_a ^ (v1857_sw & 15))) : (0)];
              float v2849_data = v37_g ? (v2849_data_pre) : (0.0f);
              float v2850_data = ir8[5];
              ir8[5] = (v2850_data + v2849_data);
              float v2856_data_pre = s0[v37_g ? ((v1863_a ^ (v1864_sw & 15))) : (0)];
              float v2856_data = v37_g ? (v2856_data_pre) : (0.0f);
              float v2857_data = ir8[6];
              ir8[6] = (v2857_data + v2856_data);
              float v2863_data_pre = s0[v37_g ? ((v1870_a ^ (v1871_sw & 15))) : (0)];
              float v2863_data = v37_g ? (v2863_data_pre) : (0.0f);
              float v2864_data = ir8[7];
              ir8[7] = (v2864_data + v2863_data);
              float v2870_data_pre = s0[v37_g ? ((v1877_a ^ (v1878_sw & 15))) : (0)];
              float v2870_data = v37_g ? (v2870_data_pre) : (0.0f);
              float v2871_data = ir8[8];
              ir8[8] = (v2871_data + v2870_data);
              float v2877_data_pre = s0[v37_g ? ((v1884_a ^ (v1885_sw & 15))) : (0)];
              float v2877_data = v37_g ? (v2877_data_pre) : (0.0f);
              float v2878_data = ir8[9];
              ir8[9] = (v2878_data + v2877_data);
              float v2884_data_pre = s0[v37_g ? ((v1891_a ^ (v1892_sw & 15))) : (0)];
              float v2884_data = v37_g ? (v2884_data_pre) : (0.0f);
              float v2885_data = ir8[10];
              ir8[10] = (v2885_data + v2884_data);
              float v2891_data_pre = s0[v37_g ? ((v1898_a ^ (v1899_sw & 15))) : (0)];
              float v2891_data = v37_g ? (v2891_data_pre) : (0.0f);
              float v2892_data = ir8[11];
              ir8[11] = (v2892_data + v2891_data);
              // r8 = ir8
              if (v37_g) {
                #pragma unroll
                for (int32_t v2894_n1 = 0; v2894_n1 < 12; ++v2894_n1) {
                  float v2896_data = ir8[v2894_n1];
                  r8[v2894_n1] = v2896_data;
                }
              }
              // glb_m5 = store{r>g}(r8);
              if (v37_g) {
                #pragma unroll
                for (int32_t v2897_i1 = 0; v2897_i1 < 12; ++v2897_i1) {
                  float v2899_data = r8[v2897_i1];
                  glb_m5[(v27_lead + (v2897_i1 * 12))] = v2899_data;
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

