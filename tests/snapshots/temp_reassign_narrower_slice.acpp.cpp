// === base name ===
kernel_6756e3259657f322

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_6756e3259657f322 = {{16, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_6756e3259657f322(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_6756e3259657f322(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_6756e3259657f322(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_6756e3259657f322(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_6756e3259657f322(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_6756e3259657f322(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_6756e3259657f322(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0) {
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
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 72 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v8_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v8_batchId0 * 24 + 0 + m4_extraOffset];
              float *const __restrict__ glb_m5 = &m5[v8_batchId0 * 144 + 0 + m5_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v25_lead = item.get_local_id(2) % 16;
              bool v26_g = v25_lead < 6;
              if (v26_g) {
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
                  float v32_data = glb_m0[(v25_lead + (v27_i1 * 6))];
                  r0[v27_i1] = v32_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              bool v35_g = v25_lead < 12;
              if (v35_g) {
                #pragma unroll
                for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
                  float v41_data = glb_m1[(v25_lead + (v36_i1 * 12))];
                  r1[v36_i1] = v41_data;
                }
              }
              float r3[12]{};
              // r3 = load{g>r}(glb_m2);
              if (v26_g) {
                #pragma unroll
                for (int32_t v919_i1 = 0; v919_i1 < 12; ++v919_i1) {
                  float v924_data = glb_m2[(v25_lead + (v919_i1 * 6))];
                  r3[v919_i1] = v924_data;
                }
              }
              float r2[12]{};
              // r2 = +(r0 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              float v44_data = r0[0];
              float v45_data = r1[0];
              float v46_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v48_data = r2[0];
              r2[0] = (v48_data + (v44_data * v46_bc));
              float v51_data = r1[1];
              float v52_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v54_data = r2[1];
              r2[1] = (v54_data + (v44_data * v52_bc));
              float v57_data = r1[2];
              float v58_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v60_data = r2[2];
              r2[2] = (v60_data + (v44_data * v58_bc));
              float v63_data = r1[3];
              float v64_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v66_data = r2[3];
              r2[3] = (v66_data + (v44_data * v64_bc));
              float v69_data = r1[4];
              float v70_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v72_data = r2[4];
              r2[4] = (v72_data + (v44_data * v70_bc));
              float v75_data = r1[5];
              float v76_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v78_data = r2[5];
              r2[5] = (v78_data + (v44_data * v76_bc));
              float v81_data = r1[6];
              float v82_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v84_data = r2[6];
              r2[6] = (v84_data + (v44_data * v82_bc));
              float v87_data = r1[7];
              float v88_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v90_data = r2[7];
              r2[7] = (v90_data + (v44_data * v88_bc));
              float v93_data = r1[8];
              float v94_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v96_data = r2[8];
              r2[8] = (v96_data + (v44_data * v94_bc));
              float v99_data = r1[9];
              float v100_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v102_data = r2[9];
              r2[9] = (v102_data + (v44_data * v100_bc));
              float v105_data = r1[10];
              float v106_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v108_data = r2[10];
              r2[10] = (v108_data + (v44_data * v106_bc));
              float v111_data = r1[11];
              float v112_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v114_data = r2[11];
              r2[11] = (v114_data + (v44_data * v112_bc));
              float v116_data = r0[1];
              float v118_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v120_data = r2[0];
              r2[0] = (v120_data + (v116_data * v118_bc));
              float v124_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v126_data = r2[1];
              r2[1] = (v126_data + (v116_data * v124_bc));
              float v130_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v132_data = r2[2];
              r2[2] = (v132_data + (v116_data * v130_bc));
              float v136_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v138_data = r2[3];
              r2[3] = (v138_data + (v116_data * v136_bc));
              float v142_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v144_data = r2[4];
              r2[4] = (v144_data + (v116_data * v142_bc));
              float v148_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v150_data = r2[5];
              r2[5] = (v150_data + (v116_data * v148_bc));
              float v154_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v156_data = r2[6];
              r2[6] = (v156_data + (v116_data * v154_bc));
              float v160_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v162_data = r2[7];
              r2[7] = (v162_data + (v116_data * v160_bc));
              float v166_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v168_data = r2[8];
              r2[8] = (v168_data + (v116_data * v166_bc));
              float v172_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v174_data = r2[9];
              r2[9] = (v174_data + (v116_data * v172_bc));
              float v178_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v180_data = r2[10];
              r2[10] = (v180_data + (v116_data * v178_bc));
              float v184_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v186_data = r2[11];
              r2[11] = (v186_data + (v116_data * v184_bc));
              float v188_data = r0[2];
              float v190_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v192_data = r2[0];
              r2[0] = (v192_data + (v188_data * v190_bc));
              float v196_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v198_data = r2[1];
              r2[1] = (v198_data + (v188_data * v196_bc));
              float v202_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v204_data = r2[2];
              r2[2] = (v204_data + (v188_data * v202_bc));
              float v208_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v210_data = r2[3];
              r2[3] = (v210_data + (v188_data * v208_bc));
              float v214_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v216_data = r2[4];
              r2[4] = (v216_data + (v188_data * v214_bc));
              float v220_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v222_data = r2[5];
              r2[5] = (v222_data + (v188_data * v220_bc));
              float v226_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v228_data = r2[6];
              r2[6] = (v228_data + (v188_data * v226_bc));
              float v232_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v234_data = r2[7];
              r2[7] = (v234_data + (v188_data * v232_bc));
              float v238_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v240_data = r2[8];
              r2[8] = (v240_data + (v188_data * v238_bc));
              float v244_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v246_data = r2[9];
              r2[9] = (v246_data + (v188_data * v244_bc));
              float v250_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v252_data = r2[10];
              r2[10] = (v252_data + (v188_data * v250_bc));
              float v256_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v258_data = r2[11];
              r2[11] = (v258_data + (v188_data * v256_bc));
              float v260_data = r0[3];
              float v262_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v264_data = r2[0];
              r2[0] = (v264_data + (v260_data * v262_bc));
              float v268_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v270_data = r2[1];
              r2[1] = (v270_data + (v260_data * v268_bc));
              float v274_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v276_data = r2[2];
              r2[2] = (v276_data + (v260_data * v274_bc));
              float v280_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v282_data = r2[3];
              r2[3] = (v282_data + (v260_data * v280_bc));
              float v286_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v288_data = r2[4];
              r2[4] = (v288_data + (v260_data * v286_bc));
              float v292_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v294_data = r2[5];
              r2[5] = (v294_data + (v260_data * v292_bc));
              float v298_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v300_data = r2[6];
              r2[6] = (v300_data + (v260_data * v298_bc));
              float v304_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v306_data = r2[7];
              r2[7] = (v306_data + (v260_data * v304_bc));
              float v310_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v312_data = r2[8];
              r2[8] = (v312_data + (v260_data * v310_bc));
              float v316_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v318_data = r2[9];
              r2[9] = (v318_data + (v260_data * v316_bc));
              float v322_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v324_data = r2[10];
              r2[10] = (v324_data + (v260_data * v322_bc));
              float v328_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v330_data = r2[11];
              r2[11] = (v330_data + (v260_data * v328_bc));
              float v332_data = r0[4];
              float v334_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v336_data = r2[0];
              r2[0] = (v336_data + (v332_data * v334_bc));
              float v340_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v342_data = r2[1];
              r2[1] = (v342_data + (v332_data * v340_bc));
              float v346_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v348_data = r2[2];
              r2[2] = (v348_data + (v332_data * v346_bc));
              float v352_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v354_data = r2[3];
              r2[3] = (v354_data + (v332_data * v352_bc));
              float v358_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v360_data = r2[4];
              r2[4] = (v360_data + (v332_data * v358_bc));
              float v364_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v366_data = r2[5];
              r2[5] = (v366_data + (v332_data * v364_bc));
              float v370_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v372_data = r2[6];
              r2[6] = (v372_data + (v332_data * v370_bc));
              float v376_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v378_data = r2[7];
              r2[7] = (v378_data + (v332_data * v376_bc));
              float v382_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v384_data = r2[8];
              r2[8] = (v384_data + (v332_data * v382_bc));
              float v388_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v390_data = r2[9];
              r2[9] = (v390_data + (v332_data * v388_bc));
              float v394_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v396_data = r2[10];
              r2[10] = (v396_data + (v332_data * v394_bc));
              float v400_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v402_data = r2[11];
              r2[11] = (v402_data + (v332_data * v400_bc));
              float v404_data = r0[5];
              float v406_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v408_data = r2[0];
              r2[0] = (v408_data + (v404_data * v406_bc));
              float v412_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v414_data = r2[1];
              r2[1] = (v414_data + (v404_data * v412_bc));
              float v418_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v420_data = r2[2];
              r2[2] = (v420_data + (v404_data * v418_bc));
              float v424_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v426_data = r2[3];
              r2[3] = (v426_data + (v404_data * v424_bc));
              float v430_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v432_data = r2[4];
              r2[4] = (v432_data + (v404_data * v430_bc));
              float v436_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v438_data = r2[5];
              r2[5] = (v438_data + (v404_data * v436_bc));
              float v442_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v444_data = r2[6];
              r2[6] = (v444_data + (v404_data * v442_bc));
              float v448_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v450_data = r2[7];
              r2[7] = (v450_data + (v404_data * v448_bc));
              float v454_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v456_data = r2[8];
              r2[8] = (v456_data + (v404_data * v454_bc));
              float v460_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v462_data = r2[9];
              r2[9] = (v462_data + (v404_data * v460_bc));
              float v466_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v468_data = r2[10];
              r2[10] = (v468_data + (v404_data * v466_bc));
              float v472_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v474_data = r2[11];
              r2[11] = (v474_data + (v404_data * v472_bc));
              float v476_data = r0[6];
              float v478_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v480_data = r2[0];
              r2[0] = (v480_data + (v476_data * v478_bc));
              float v484_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v486_data = r2[1];
              r2[1] = (v486_data + (v476_data * v484_bc));
              float v490_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v492_data = r2[2];
              r2[2] = (v492_data + (v476_data * v490_bc));
              float v496_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v498_data = r2[3];
              r2[3] = (v498_data + (v476_data * v496_bc));
              float v502_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v504_data = r2[4];
              r2[4] = (v504_data + (v476_data * v502_bc));
              float v508_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v510_data = r2[5];
              r2[5] = (v510_data + (v476_data * v508_bc));
              float v514_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v516_data = r2[6];
              r2[6] = (v516_data + (v476_data * v514_bc));
              float v520_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v522_data = r2[7];
              r2[7] = (v522_data + (v476_data * v520_bc));
              float v526_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v528_data = r2[8];
              r2[8] = (v528_data + (v476_data * v526_bc));
              float v532_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v534_data = r2[9];
              r2[9] = (v534_data + (v476_data * v532_bc));
              float v538_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v540_data = r2[10];
              r2[10] = (v540_data + (v476_data * v538_bc));
              float v544_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v546_data = r2[11];
              r2[11] = (v546_data + (v476_data * v544_bc));
              float v548_data = r0[7];
              float v550_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v552_data = r2[0];
              r2[0] = (v552_data + (v548_data * v550_bc));
              float v556_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v558_data = r2[1];
              r2[1] = (v558_data + (v548_data * v556_bc));
              float v562_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v564_data = r2[2];
              r2[2] = (v564_data + (v548_data * v562_bc));
              float v568_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v570_data = r2[3];
              r2[3] = (v570_data + (v548_data * v568_bc));
              float v574_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v576_data = r2[4];
              r2[4] = (v576_data + (v548_data * v574_bc));
              float v580_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v582_data = r2[5];
              r2[5] = (v582_data + (v548_data * v580_bc));
              float v586_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v588_data = r2[6];
              r2[6] = (v588_data + (v548_data * v586_bc));
              float v592_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v594_data = r2[7];
              r2[7] = (v594_data + (v548_data * v592_bc));
              float v598_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v600_data = r2[8];
              r2[8] = (v600_data + (v548_data * v598_bc));
              float v604_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v606_data = r2[9];
              r2[9] = (v606_data + (v548_data * v604_bc));
              float v610_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v612_data = r2[10];
              r2[10] = (v612_data + (v548_data * v610_bc));
              float v616_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v618_data = r2[11];
              r2[11] = (v618_data + (v548_data * v616_bc));
              float v620_data = r0[8];
              float v622_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v624_data = r2[0];
              r2[0] = (v624_data + (v620_data * v622_bc));
              float v628_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v630_data = r2[1];
              r2[1] = (v630_data + (v620_data * v628_bc));
              float v634_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v636_data = r2[2];
              r2[2] = (v636_data + (v620_data * v634_bc));
              float v640_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v642_data = r2[3];
              r2[3] = (v642_data + (v620_data * v640_bc));
              float v646_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v648_data = r2[4];
              r2[4] = (v648_data + (v620_data * v646_bc));
              float v652_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v654_data = r2[5];
              r2[5] = (v654_data + (v620_data * v652_bc));
              float v658_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v660_data = r2[6];
              r2[6] = (v660_data + (v620_data * v658_bc));
              float v664_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v666_data = r2[7];
              r2[7] = (v666_data + (v620_data * v664_bc));
              float v670_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v672_data = r2[8];
              r2[8] = (v672_data + (v620_data * v670_bc));
              float v676_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v678_data = r2[9];
              r2[9] = (v678_data + (v620_data * v676_bc));
              float v682_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v684_data = r2[10];
              r2[10] = (v684_data + (v620_data * v682_bc));
              float v688_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v690_data = r2[11];
              r2[11] = (v690_data + (v620_data * v688_bc));
              float v692_data = r0[9];
              float v694_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v696_data = r2[0];
              r2[0] = (v696_data + (v692_data * v694_bc));
              float v700_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v702_data = r2[1];
              r2[1] = (v702_data + (v692_data * v700_bc));
              float v706_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v708_data = r2[2];
              r2[2] = (v708_data + (v692_data * v706_bc));
              float v712_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v714_data = r2[3];
              r2[3] = (v714_data + (v692_data * v712_bc));
              float v718_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v720_data = r2[4];
              r2[4] = (v720_data + (v692_data * v718_bc));
              float v724_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v726_data = r2[5];
              r2[5] = (v726_data + (v692_data * v724_bc));
              float v730_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v732_data = r2[6];
              r2[6] = (v732_data + (v692_data * v730_bc));
              float v736_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v738_data = r2[7];
              r2[7] = (v738_data + (v692_data * v736_bc));
              float v742_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v744_data = r2[8];
              r2[8] = (v744_data + (v692_data * v742_bc));
              float v748_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v750_data = r2[9];
              r2[9] = (v750_data + (v692_data * v748_bc));
              float v754_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v756_data = r2[10];
              r2[10] = (v756_data + (v692_data * v754_bc));
              float v760_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v762_data = r2[11];
              r2[11] = (v762_data + (v692_data * v760_bc));
              float v764_data = r0[10];
              float v766_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v768_data = r2[0];
              r2[0] = (v768_data + (v764_data * v766_bc));
              float v772_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v774_data = r2[1];
              r2[1] = (v774_data + (v764_data * v772_bc));
              float v778_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v780_data = r2[2];
              r2[2] = (v780_data + (v764_data * v778_bc));
              float v784_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v786_data = r2[3];
              r2[3] = (v786_data + (v764_data * v784_bc));
              float v790_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v792_data = r2[4];
              r2[4] = (v792_data + (v764_data * v790_bc));
              float v796_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v798_data = r2[5];
              r2[5] = (v798_data + (v764_data * v796_bc));
              float v802_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v804_data = r2[6];
              r2[6] = (v804_data + (v764_data * v802_bc));
              float v808_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v810_data = r2[7];
              r2[7] = (v810_data + (v764_data * v808_bc));
              float v814_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v816_data = r2[8];
              r2[8] = (v816_data + (v764_data * v814_bc));
              float v820_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v822_data = r2[9];
              r2[9] = (v822_data + (v764_data * v820_bc));
              float v826_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v828_data = r2[10];
              r2[10] = (v828_data + (v764_data * v826_bc));
              float v832_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v834_data = r2[11];
              r2[11] = (v834_data + (v764_data * v832_bc));
              float v836_data = r0[11];
              float v838_bc = sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v840_data = r2[0];
              r2[0] = (v840_data + (v836_data * v838_bc));
              float v844_bc = sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v846_data = r2[1];
              r2[1] = (v846_data + (v836_data * v844_bc));
              float v850_bc = sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v852_data = r2[2];
              r2[2] = (v852_data + (v836_data * v850_bc));
              float v856_bc = sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v858_data = r2[3];
              r2[3] = (v858_data + (v836_data * v856_bc));
              float v862_bc = sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v864_data = r2[4];
              r2[4] = (v864_data + (v836_data * v862_bc));
              float v868_bc = sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v870_data = r2[5];
              r2[5] = (v870_data + (v836_data * v868_bc));
              float v874_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v876_data = r2[6];
              r2[6] = (v876_data + (v836_data * v874_bc));
              float v880_bc = sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v882_data = r2[7];
              r2[7] = (v882_data + (v836_data * v880_bc));
              float v886_bc = sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v888_data = r2[8];
              r2[8] = (v888_data + (v836_data * v886_bc));
              float v892_bc = sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v894_data = r2[9];
              r2[9] = (v894_data + (v836_data * v892_bc));
              float v898_bc = sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v900_data = r2[10];
              r2[10] = (v900_data + (v836_data * v898_bc));
              float v904_bc = sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v906_data = r2[11];
              r2[11] = (v906_data + (v836_data * v904_bc));
              // s0 = store{r>s}(localShrMem0, r2);
              if (v26_g) {
                #pragma unroll
                for (int32_t v908_i1 = 0; v908_i1 < 12; ++v908_i1) {
                  float v910_data = r2[v908_i1];
                  int32_t v914_a = v25_lead + (v908_i1 * 12);
                  s0[(v914_a ^ ((v914_a >> 4) & 15))] = v910_data;
                }
              }
              float r6[12]{};
              // r6 = load{g>r}(glb_m4);
              bool v1905_g = v25_lead < 2;
              if (v1905_g) {
                #pragma unroll
                for (int32_t v1906_i1 = 0; v1906_i1 < 12; ++v1906_i1) {
                  float v1911_data = glb_m4[(v25_lead + (v1906_i1 * 2))];
                  r6[v1906_i1] = v1911_data;
                }
              }
              float r4[12]{};
              // ir4 = +(r3 * r1)
              // [(0, 6), (0, 12)] [(0, 12)]
              float ir4[12]{};
              float v928_data = r3[0];
              float v932_data = ir4[0];
              ir4[0] = (v932_data + (v928_data * v46_bc));
              float v938_data = ir4[1];
              ir4[1] = (v938_data + (v928_data * v52_bc));
              float v944_data = ir4[2];
              ir4[2] = (v944_data + (v928_data * v58_bc));
              float v950_data = ir4[3];
              ir4[3] = (v950_data + (v928_data * v64_bc));
              float v956_data = ir4[4];
              ir4[4] = (v956_data + (v928_data * v70_bc));
              float v962_data = ir4[5];
              ir4[5] = (v962_data + (v928_data * v76_bc));
              float v968_data = ir4[6];
              ir4[6] = (v968_data + (v928_data * v82_bc));
              float v974_data = ir4[7];
              ir4[7] = (v974_data + (v928_data * v88_bc));
              float v980_data = ir4[8];
              ir4[8] = (v980_data + (v928_data * v94_bc));
              float v986_data = ir4[9];
              ir4[9] = (v986_data + (v928_data * v100_bc));
              float v992_data = ir4[10];
              ir4[10] = (v992_data + (v928_data * v106_bc));
              float v998_data = ir4[11];
              ir4[11] = (v998_data + (v928_data * v112_bc));
              float v1000_data = r3[1];
              float v1004_data = ir4[0];
              ir4[0] = (v1004_data + (v1000_data * v118_bc));
              float v1010_data = ir4[1];
              ir4[1] = (v1010_data + (v1000_data * v124_bc));
              float v1016_data = ir4[2];
              ir4[2] = (v1016_data + (v1000_data * v130_bc));
              float v1022_data = ir4[3];
              ir4[3] = (v1022_data + (v1000_data * v136_bc));
              float v1028_data = ir4[4];
              ir4[4] = (v1028_data + (v1000_data * v142_bc));
              float v1034_data = ir4[5];
              ir4[5] = (v1034_data + (v1000_data * v148_bc));
              float v1040_data = ir4[6];
              ir4[6] = (v1040_data + (v1000_data * v154_bc));
              float v1046_data = ir4[7];
              ir4[7] = (v1046_data + (v1000_data * v160_bc));
              float v1052_data = ir4[8];
              ir4[8] = (v1052_data + (v1000_data * v166_bc));
              float v1058_data = ir4[9];
              ir4[9] = (v1058_data + (v1000_data * v172_bc));
              float v1064_data = ir4[10];
              ir4[10] = (v1064_data + (v1000_data * v178_bc));
              float v1070_data = ir4[11];
              ir4[11] = (v1070_data + (v1000_data * v184_bc));
              float v1072_data = r3[2];
              float v1076_data = ir4[0];
              ir4[0] = (v1076_data + (v1072_data * v190_bc));
              float v1082_data = ir4[1];
              ir4[1] = (v1082_data + (v1072_data * v196_bc));
              float v1088_data = ir4[2];
              ir4[2] = (v1088_data + (v1072_data * v202_bc));
              float v1094_data = ir4[3];
              ir4[3] = (v1094_data + (v1072_data * v208_bc));
              float v1100_data = ir4[4];
              ir4[4] = (v1100_data + (v1072_data * v214_bc));
              float v1106_data = ir4[5];
              ir4[5] = (v1106_data + (v1072_data * v220_bc));
              float v1112_data = ir4[6];
              ir4[6] = (v1112_data + (v1072_data * v226_bc));
              float v1118_data = ir4[7];
              ir4[7] = (v1118_data + (v1072_data * v232_bc));
              float v1124_data = ir4[8];
              ir4[8] = (v1124_data + (v1072_data * v238_bc));
              float v1130_data = ir4[9];
              ir4[9] = (v1130_data + (v1072_data * v244_bc));
              float v1136_data = ir4[10];
              ir4[10] = (v1136_data + (v1072_data * v250_bc));
              float v1142_data = ir4[11];
              ir4[11] = (v1142_data + (v1072_data * v256_bc));
              float v1144_data = r3[3];
              float v1148_data = ir4[0];
              ir4[0] = (v1148_data + (v1144_data * v262_bc));
              float v1154_data = ir4[1];
              ir4[1] = (v1154_data + (v1144_data * v268_bc));
              float v1160_data = ir4[2];
              ir4[2] = (v1160_data + (v1144_data * v274_bc));
              float v1166_data = ir4[3];
              ir4[3] = (v1166_data + (v1144_data * v280_bc));
              float v1172_data = ir4[4];
              ir4[4] = (v1172_data + (v1144_data * v286_bc));
              float v1178_data = ir4[5];
              ir4[5] = (v1178_data + (v1144_data * v292_bc));
              float v1184_data = ir4[6];
              ir4[6] = (v1184_data + (v1144_data * v298_bc));
              float v1190_data = ir4[7];
              ir4[7] = (v1190_data + (v1144_data * v304_bc));
              float v1196_data = ir4[8];
              ir4[8] = (v1196_data + (v1144_data * v310_bc));
              float v1202_data = ir4[9];
              ir4[9] = (v1202_data + (v1144_data * v316_bc));
              float v1208_data = ir4[10];
              ir4[10] = (v1208_data + (v1144_data * v322_bc));
              float v1214_data = ir4[11];
              ir4[11] = (v1214_data + (v1144_data * v328_bc));
              float v1216_data = r3[4];
              float v1220_data = ir4[0];
              ir4[0] = (v1220_data + (v1216_data * v334_bc));
              float v1226_data = ir4[1];
              ir4[1] = (v1226_data + (v1216_data * v340_bc));
              float v1232_data = ir4[2];
              ir4[2] = (v1232_data + (v1216_data * v346_bc));
              float v1238_data = ir4[3];
              ir4[3] = (v1238_data + (v1216_data * v352_bc));
              float v1244_data = ir4[4];
              ir4[4] = (v1244_data + (v1216_data * v358_bc));
              float v1250_data = ir4[5];
              ir4[5] = (v1250_data + (v1216_data * v364_bc));
              float v1256_data = ir4[6];
              ir4[6] = (v1256_data + (v1216_data * v370_bc));
              float v1262_data = ir4[7];
              ir4[7] = (v1262_data + (v1216_data * v376_bc));
              float v1268_data = ir4[8];
              ir4[8] = (v1268_data + (v1216_data * v382_bc));
              float v1274_data = ir4[9];
              ir4[9] = (v1274_data + (v1216_data * v388_bc));
              float v1280_data = ir4[10];
              ir4[10] = (v1280_data + (v1216_data * v394_bc));
              float v1286_data = ir4[11];
              ir4[11] = (v1286_data + (v1216_data * v400_bc));
              float v1288_data = r3[5];
              float v1292_data = ir4[0];
              ir4[0] = (v1292_data + (v1288_data * v406_bc));
              float v1298_data = ir4[1];
              ir4[1] = (v1298_data + (v1288_data * v412_bc));
              float v1304_data = ir4[2];
              ir4[2] = (v1304_data + (v1288_data * v418_bc));
              float v1310_data = ir4[3];
              ir4[3] = (v1310_data + (v1288_data * v424_bc));
              float v1316_data = ir4[4];
              ir4[4] = (v1316_data + (v1288_data * v430_bc));
              float v1322_data = ir4[5];
              ir4[5] = (v1322_data + (v1288_data * v436_bc));
              float v1328_data = ir4[6];
              ir4[6] = (v1328_data + (v1288_data * v442_bc));
              float v1334_data = ir4[7];
              ir4[7] = (v1334_data + (v1288_data * v448_bc));
              float v1340_data = ir4[8];
              ir4[8] = (v1340_data + (v1288_data * v454_bc));
              float v1346_data = ir4[9];
              ir4[9] = (v1346_data + (v1288_data * v460_bc));
              float v1352_data = ir4[10];
              ir4[10] = (v1352_data + (v1288_data * v466_bc));
              float v1358_data = ir4[11];
              ir4[11] = (v1358_data + (v1288_data * v472_bc));
              float v1360_data = r3[6];
              float v1364_data = ir4[0];
              ir4[0] = (v1364_data + (v1360_data * v478_bc));
              float v1370_data = ir4[1];
              ir4[1] = (v1370_data + (v1360_data * v484_bc));
              float v1376_data = ir4[2];
              ir4[2] = (v1376_data + (v1360_data * v490_bc));
              float v1382_data = ir4[3];
              ir4[3] = (v1382_data + (v1360_data * v496_bc));
              float v1388_data = ir4[4];
              ir4[4] = (v1388_data + (v1360_data * v502_bc));
              float v1394_data = ir4[5];
              ir4[5] = (v1394_data + (v1360_data * v508_bc));
              float v1400_data = ir4[6];
              ir4[6] = (v1400_data + (v1360_data * v514_bc));
              float v1406_data = ir4[7];
              ir4[7] = (v1406_data + (v1360_data * v520_bc));
              float v1412_data = ir4[8];
              ir4[8] = (v1412_data + (v1360_data * v526_bc));
              float v1418_data = ir4[9];
              ir4[9] = (v1418_data + (v1360_data * v532_bc));
              float v1424_data = ir4[10];
              ir4[10] = (v1424_data + (v1360_data * v538_bc));
              float v1430_data = ir4[11];
              ir4[11] = (v1430_data + (v1360_data * v544_bc));
              float v1432_data = r3[7];
              float v1436_data = ir4[0];
              ir4[0] = (v1436_data + (v1432_data * v550_bc));
              float v1442_data = ir4[1];
              ir4[1] = (v1442_data + (v1432_data * v556_bc));
              float v1448_data = ir4[2];
              ir4[2] = (v1448_data + (v1432_data * v562_bc));
              float v1454_data = ir4[3];
              ir4[3] = (v1454_data + (v1432_data * v568_bc));
              float v1460_data = ir4[4];
              ir4[4] = (v1460_data + (v1432_data * v574_bc));
              float v1466_data = ir4[5];
              ir4[5] = (v1466_data + (v1432_data * v580_bc));
              float v1472_data = ir4[6];
              ir4[6] = (v1472_data + (v1432_data * v586_bc));
              float v1478_data = ir4[7];
              ir4[7] = (v1478_data + (v1432_data * v592_bc));
              float v1484_data = ir4[8];
              ir4[8] = (v1484_data + (v1432_data * v598_bc));
              float v1490_data = ir4[9];
              ir4[9] = (v1490_data + (v1432_data * v604_bc));
              float v1496_data = ir4[10];
              ir4[10] = (v1496_data + (v1432_data * v610_bc));
              float v1502_data = ir4[11];
              ir4[11] = (v1502_data + (v1432_data * v616_bc));
              float v1504_data = r3[8];
              float v1508_data = ir4[0];
              ir4[0] = (v1508_data + (v1504_data * v622_bc));
              float v1514_data = ir4[1];
              ir4[1] = (v1514_data + (v1504_data * v628_bc));
              float v1520_data = ir4[2];
              ir4[2] = (v1520_data + (v1504_data * v634_bc));
              float v1526_data = ir4[3];
              ir4[3] = (v1526_data + (v1504_data * v640_bc));
              float v1532_data = ir4[4];
              ir4[4] = (v1532_data + (v1504_data * v646_bc));
              float v1538_data = ir4[5];
              ir4[5] = (v1538_data + (v1504_data * v652_bc));
              float v1544_data = ir4[6];
              ir4[6] = (v1544_data + (v1504_data * v658_bc));
              float v1550_data = ir4[7];
              ir4[7] = (v1550_data + (v1504_data * v664_bc));
              float v1556_data = ir4[8];
              ir4[8] = (v1556_data + (v1504_data * v670_bc));
              float v1562_data = ir4[9];
              ir4[9] = (v1562_data + (v1504_data * v676_bc));
              float v1568_data = ir4[10];
              ir4[10] = (v1568_data + (v1504_data * v682_bc));
              float v1574_data = ir4[11];
              ir4[11] = (v1574_data + (v1504_data * v688_bc));
              float v1576_data = r3[9];
              float v1580_data = ir4[0];
              ir4[0] = (v1580_data + (v1576_data * v694_bc));
              float v1586_data = ir4[1];
              ir4[1] = (v1586_data + (v1576_data * v700_bc));
              float v1592_data = ir4[2];
              ir4[2] = (v1592_data + (v1576_data * v706_bc));
              float v1598_data = ir4[3];
              ir4[3] = (v1598_data + (v1576_data * v712_bc));
              float v1604_data = ir4[4];
              ir4[4] = (v1604_data + (v1576_data * v718_bc));
              float v1610_data = ir4[5];
              ir4[5] = (v1610_data + (v1576_data * v724_bc));
              float v1616_data = ir4[6];
              ir4[6] = (v1616_data + (v1576_data * v730_bc));
              float v1622_data = ir4[7];
              ir4[7] = (v1622_data + (v1576_data * v736_bc));
              float v1628_data = ir4[8];
              ir4[8] = (v1628_data + (v1576_data * v742_bc));
              float v1634_data = ir4[9];
              ir4[9] = (v1634_data + (v1576_data * v748_bc));
              float v1640_data = ir4[10];
              ir4[10] = (v1640_data + (v1576_data * v754_bc));
              float v1646_data = ir4[11];
              ir4[11] = (v1646_data + (v1576_data * v760_bc));
              float v1648_data = r3[10];
              float v1652_data = ir4[0];
              ir4[0] = (v1652_data + (v1648_data * v766_bc));
              float v1658_data = ir4[1];
              ir4[1] = (v1658_data + (v1648_data * v772_bc));
              float v1664_data = ir4[2];
              ir4[2] = (v1664_data + (v1648_data * v778_bc));
              float v1670_data = ir4[3];
              ir4[3] = (v1670_data + (v1648_data * v784_bc));
              float v1676_data = ir4[4];
              ir4[4] = (v1676_data + (v1648_data * v790_bc));
              float v1682_data = ir4[5];
              ir4[5] = (v1682_data + (v1648_data * v796_bc));
              float v1688_data = ir4[6];
              ir4[6] = (v1688_data + (v1648_data * v802_bc));
              float v1694_data = ir4[7];
              ir4[7] = (v1694_data + (v1648_data * v808_bc));
              float v1700_data = ir4[8];
              ir4[8] = (v1700_data + (v1648_data * v814_bc));
              float v1706_data = ir4[9];
              ir4[9] = (v1706_data + (v1648_data * v820_bc));
              float v1712_data = ir4[10];
              ir4[10] = (v1712_data + (v1648_data * v826_bc));
              float v1718_data = ir4[11];
              ir4[11] = (v1718_data + (v1648_data * v832_bc));
              float v1720_data = r3[11];
              float v1724_data = ir4[0];
              ir4[0] = (v1724_data + (v1720_data * v838_bc));
              float v1730_data = ir4[1];
              ir4[1] = (v1730_data + (v1720_data * v844_bc));
              float v1736_data = ir4[2];
              ir4[2] = (v1736_data + (v1720_data * v850_bc));
              float v1742_data = ir4[3];
              ir4[3] = (v1742_data + (v1720_data * v856_bc));
              float v1748_data = ir4[4];
              ir4[4] = (v1748_data + (v1720_data * v862_bc));
              float v1754_data = ir4[5];
              ir4[5] = (v1754_data + (v1720_data * v868_bc));
              float v1760_data = ir4[6];
              ir4[6] = (v1760_data + (v1720_data * v874_bc));
              float v1766_data = ir4[7];
              ir4[7] = (v1766_data + (v1720_data * v880_bc));
              float v1772_data = ir4[8];
              ir4[8] = (v1772_data + (v1720_data * v886_bc));
              float v1778_data = ir4[9];
              ir4[9] = (v1778_data + (v1720_data * v892_bc));
              float v1784_data = ir4[10];
              ir4[10] = (v1784_data + (v1720_data * v898_bc));
              float v1790_data = ir4[11];
              ir4[11] = (v1790_data + (v1720_data * v904_bc));
              // r4 = ir4
              if (v26_g) {
                #pragma unroll
                for (int32_t v1792_n1 = 0; v1792_n1 < 12; ++v1792_n1) {
                  float v1794_data = ir4[v1792_n1];
                  r4[v1792_n1] = v1794_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r4);
              if (v26_g) {
                int32_t v1800_off = v25_lead + 6;
                #pragma unroll
                for (int32_t v1795_i1 = 0; v1795_i1 < 12; ++v1795_i1) {
                  float v1797_data = r4[v1795_i1];
                  int32_t v1802_a = v1800_off + (v1795_i1 * 12);
                  s0[(v1802_a ^ ((v1802_a >> 4) & 15))] = v1797_data;
                }
              }
              float r5[12]{};
              // ir5 = +(s0)
              // [(0, 12), (0, 12)] []
              float ir5[12]{};
              int32_t v1812_sw = (v25_lead >> 4) & 15;
              int32_t v1813_sw = v25_lead ^ v1812_sw;
              sycl::group_barrier(item.get_sub_group());
              float v1814_data_pre = s0[v35_g ? (v1813_sw) : (0)];
              float v1814_data = v35_g ? (v1814_data_pre) : (0.0f);
              float v1815_data = ir5[0];
              ir5[0] = (v1815_data + v1814_data);
              int32_t v1817_a = v25_lead + 12;
              int32_t v1818_sw = v1817_a >> 4;
              float v1821_data_pre = s0[v35_g ? ((v1817_a ^ (v1818_sw & 15))) : (0)];
              float v1821_data = v35_g ? (v1821_data_pre) : (0.0f);
              float v1822_data = ir5[1];
              ir5[1] = (v1822_data + v1821_data);
              int32_t v1824_a = v25_lead + 24;
              int32_t v1825_sw = v1824_a >> 4;
              float v1828_data_pre = s0[v35_g ? ((v1824_a ^ (v1825_sw & 15))) : (0)];
              float v1828_data = v35_g ? (v1828_data_pre) : (0.0f);
              float v1829_data = ir5[2];
              ir5[2] = (v1829_data + v1828_data);
              int32_t v1831_a = v25_lead + 36;
              int32_t v1832_sw = v1831_a >> 4;
              float v1835_data_pre = s0[v35_g ? ((v1831_a ^ (v1832_sw & 15))) : (0)];
              float v1835_data = v35_g ? (v1835_data_pre) : (0.0f);
              float v1836_data = ir5[3];
              ir5[3] = (v1836_data + v1835_data);
              int32_t v1838_a = v25_lead + 48;
              int32_t v1839_sw = v1838_a >> 4;
              float v1842_data_pre = s0[v35_g ? ((v1838_a ^ (v1839_sw & 15))) : (0)];
              float v1842_data = v35_g ? (v1842_data_pre) : (0.0f);
              float v1843_data = ir5[4];
              ir5[4] = (v1843_data + v1842_data);
              int32_t v1845_a = v25_lead + 60;
              int32_t v1846_sw = v1845_a >> 4;
              float v1849_data_pre = s0[v35_g ? ((v1845_a ^ (v1846_sw & 15))) : (0)];
              float v1849_data = v35_g ? (v1849_data_pre) : (0.0f);
              float v1850_data = ir5[5];
              ir5[5] = (v1850_data + v1849_data);
              int32_t v1852_a = v25_lead + 72;
              int32_t v1853_sw = v1852_a >> 4;
              float v1856_data_pre = s0[v35_g ? ((v1852_a ^ (v1853_sw & 15))) : (0)];
              float v1856_data = v35_g ? (v1856_data_pre) : (0.0f);
              float v1857_data = ir5[6];
              ir5[6] = (v1857_data + v1856_data);
              int32_t v1859_a = v25_lead + 84;
              int32_t v1860_sw = v1859_a >> 4;
              float v1863_data_pre = s0[v35_g ? ((v1859_a ^ (v1860_sw & 15))) : (0)];
              float v1863_data = v35_g ? (v1863_data_pre) : (0.0f);
              float v1864_data = ir5[7];
              ir5[7] = (v1864_data + v1863_data);
              int32_t v1866_a = v25_lead + 96;
              int32_t v1867_sw = v1866_a >> 4;
              float v1870_data_pre = s0[v35_g ? ((v1866_a ^ (v1867_sw & 15))) : (0)];
              float v1870_data = v35_g ? (v1870_data_pre) : (0.0f);
              float v1871_data = ir5[8];
              ir5[8] = (v1871_data + v1870_data);
              int32_t v1873_a = v25_lead + 108;
              int32_t v1874_sw = v1873_a >> 4;
              float v1877_data_pre = s0[v35_g ? ((v1873_a ^ (v1874_sw & 15))) : (0)];
              float v1877_data = v35_g ? (v1877_data_pre) : (0.0f);
              float v1878_data = ir5[9];
              ir5[9] = (v1878_data + v1877_data);
              int32_t v1880_a = v25_lead + 120;
              int32_t v1881_sw = v1880_a >> 4;
              float v1884_data_pre = s0[v35_g ? ((v1880_a ^ (v1881_sw & 15))) : (0)];
              float v1884_data = v35_g ? (v1884_data_pre) : (0.0f);
              float v1885_data = ir5[10];
              ir5[10] = (v1885_data + v1884_data);
              int32_t v1887_a = v25_lead + 132;
              int32_t v1888_sw = v1887_a >> 4;
              float v1891_data_pre = s0[v35_g ? ((v1887_a ^ (v1888_sw & 15))) : (0)];
              float v1891_data = v35_g ? (v1891_data_pre) : (0.0f);
              float v1892_data = ir5[11];
              ir5[11] = (v1892_data + v1891_data);
              // r5 = ir5
              if (v35_g) {
                #pragma unroll
                for (int32_t v1894_n1 = 0; v1894_n1 < 12; ++v1894_n1) {
                  float v1896_data = ir5[v1894_n1];
                  r5[v1894_n1] = v1896_data;
                }
              }
              // glb_m3 = store{r>g}(r5);
              if (v35_g) {
                #pragma unroll
                for (int32_t v1897_i1 = 0; v1897_i1 < 12; ++v1897_i1) {
                  float v1899_data = r5[v1897_i1];
                  glb_m3[(v25_lead + (v1897_i1 * 12))] = v1899_data;
                }
              }
              float r7[12]{};
              // ir7 = +(r6 * r1)
              // [(0, 2), (0, 12)] [(0, 12)]
              float ir7[12]{};
              float v1915_data = r6[0];
              float v1919_data = ir7[0];
              ir7[0] = (v1919_data + (v1915_data * v46_bc));
              float v1925_data = ir7[1];
              ir7[1] = (v1925_data + (v1915_data * v52_bc));
              float v1931_data = ir7[2];
              ir7[2] = (v1931_data + (v1915_data * v58_bc));
              float v1937_data = ir7[3];
              ir7[3] = (v1937_data + (v1915_data * v64_bc));
              float v1943_data = ir7[4];
              ir7[4] = (v1943_data + (v1915_data * v70_bc));
              float v1949_data = ir7[5];
              ir7[5] = (v1949_data + (v1915_data * v76_bc));
              float v1955_data = ir7[6];
              ir7[6] = (v1955_data + (v1915_data * v82_bc));
              float v1961_data = ir7[7];
              ir7[7] = (v1961_data + (v1915_data * v88_bc));
              float v1967_data = ir7[8];
              ir7[8] = (v1967_data + (v1915_data * v94_bc));
              float v1973_data = ir7[9];
              ir7[9] = (v1973_data + (v1915_data * v100_bc));
              float v1979_data = ir7[10];
              ir7[10] = (v1979_data + (v1915_data * v106_bc));
              float v1985_data = ir7[11];
              ir7[11] = (v1985_data + (v1915_data * v112_bc));
              float v1987_data = r6[1];
              float v1991_data = ir7[0];
              ir7[0] = (v1991_data + (v1987_data * v118_bc));
              float v1997_data = ir7[1];
              ir7[1] = (v1997_data + (v1987_data * v124_bc));
              float v2003_data = ir7[2];
              ir7[2] = (v2003_data + (v1987_data * v130_bc));
              float v2009_data = ir7[3];
              ir7[3] = (v2009_data + (v1987_data * v136_bc));
              float v2015_data = ir7[4];
              ir7[4] = (v2015_data + (v1987_data * v142_bc));
              float v2021_data = ir7[5];
              ir7[5] = (v2021_data + (v1987_data * v148_bc));
              float v2027_data = ir7[6];
              ir7[6] = (v2027_data + (v1987_data * v154_bc));
              float v2033_data = ir7[7];
              ir7[7] = (v2033_data + (v1987_data * v160_bc));
              float v2039_data = ir7[8];
              ir7[8] = (v2039_data + (v1987_data * v166_bc));
              float v2045_data = ir7[9];
              ir7[9] = (v2045_data + (v1987_data * v172_bc));
              float v2051_data = ir7[10];
              ir7[10] = (v2051_data + (v1987_data * v178_bc));
              float v2057_data = ir7[11];
              ir7[11] = (v2057_data + (v1987_data * v184_bc));
              float v2059_data = r6[2];
              float v2063_data = ir7[0];
              ir7[0] = (v2063_data + (v2059_data * v190_bc));
              float v2069_data = ir7[1];
              ir7[1] = (v2069_data + (v2059_data * v196_bc));
              float v2075_data = ir7[2];
              ir7[2] = (v2075_data + (v2059_data * v202_bc));
              float v2081_data = ir7[3];
              ir7[3] = (v2081_data + (v2059_data * v208_bc));
              float v2087_data = ir7[4];
              ir7[4] = (v2087_data + (v2059_data * v214_bc));
              float v2093_data = ir7[5];
              ir7[5] = (v2093_data + (v2059_data * v220_bc));
              float v2099_data = ir7[6];
              ir7[6] = (v2099_data + (v2059_data * v226_bc));
              float v2105_data = ir7[7];
              ir7[7] = (v2105_data + (v2059_data * v232_bc));
              float v2111_data = ir7[8];
              ir7[8] = (v2111_data + (v2059_data * v238_bc));
              float v2117_data = ir7[9];
              ir7[9] = (v2117_data + (v2059_data * v244_bc));
              float v2123_data = ir7[10];
              ir7[10] = (v2123_data + (v2059_data * v250_bc));
              float v2129_data = ir7[11];
              ir7[11] = (v2129_data + (v2059_data * v256_bc));
              float v2131_data = r6[3];
              float v2135_data = ir7[0];
              ir7[0] = (v2135_data + (v2131_data * v262_bc));
              float v2141_data = ir7[1];
              ir7[1] = (v2141_data + (v2131_data * v268_bc));
              float v2147_data = ir7[2];
              ir7[2] = (v2147_data + (v2131_data * v274_bc));
              float v2153_data = ir7[3];
              ir7[3] = (v2153_data + (v2131_data * v280_bc));
              float v2159_data = ir7[4];
              ir7[4] = (v2159_data + (v2131_data * v286_bc));
              float v2165_data = ir7[5];
              ir7[5] = (v2165_data + (v2131_data * v292_bc));
              float v2171_data = ir7[6];
              ir7[6] = (v2171_data + (v2131_data * v298_bc));
              float v2177_data = ir7[7];
              ir7[7] = (v2177_data + (v2131_data * v304_bc));
              float v2183_data = ir7[8];
              ir7[8] = (v2183_data + (v2131_data * v310_bc));
              float v2189_data = ir7[9];
              ir7[9] = (v2189_data + (v2131_data * v316_bc));
              float v2195_data = ir7[10];
              ir7[10] = (v2195_data + (v2131_data * v322_bc));
              float v2201_data = ir7[11];
              ir7[11] = (v2201_data + (v2131_data * v328_bc));
              float v2203_data = r6[4];
              float v2207_data = ir7[0];
              ir7[0] = (v2207_data + (v2203_data * v334_bc));
              float v2213_data = ir7[1];
              ir7[1] = (v2213_data + (v2203_data * v340_bc));
              float v2219_data = ir7[2];
              ir7[2] = (v2219_data + (v2203_data * v346_bc));
              float v2225_data = ir7[3];
              ir7[3] = (v2225_data + (v2203_data * v352_bc));
              float v2231_data = ir7[4];
              ir7[4] = (v2231_data + (v2203_data * v358_bc));
              float v2237_data = ir7[5];
              ir7[5] = (v2237_data + (v2203_data * v364_bc));
              float v2243_data = ir7[6];
              ir7[6] = (v2243_data + (v2203_data * v370_bc));
              float v2249_data = ir7[7];
              ir7[7] = (v2249_data + (v2203_data * v376_bc));
              float v2255_data = ir7[8];
              ir7[8] = (v2255_data + (v2203_data * v382_bc));
              float v2261_data = ir7[9];
              ir7[9] = (v2261_data + (v2203_data * v388_bc));
              float v2267_data = ir7[10];
              ir7[10] = (v2267_data + (v2203_data * v394_bc));
              float v2273_data = ir7[11];
              ir7[11] = (v2273_data + (v2203_data * v400_bc));
              float v2275_data = r6[5];
              float v2279_data = ir7[0];
              ir7[0] = (v2279_data + (v2275_data * v406_bc));
              float v2285_data = ir7[1];
              ir7[1] = (v2285_data + (v2275_data * v412_bc));
              float v2291_data = ir7[2];
              ir7[2] = (v2291_data + (v2275_data * v418_bc));
              float v2297_data = ir7[3];
              ir7[3] = (v2297_data + (v2275_data * v424_bc));
              float v2303_data = ir7[4];
              ir7[4] = (v2303_data + (v2275_data * v430_bc));
              float v2309_data = ir7[5];
              ir7[5] = (v2309_data + (v2275_data * v436_bc));
              float v2315_data = ir7[6];
              ir7[6] = (v2315_data + (v2275_data * v442_bc));
              float v2321_data = ir7[7];
              ir7[7] = (v2321_data + (v2275_data * v448_bc));
              float v2327_data = ir7[8];
              ir7[8] = (v2327_data + (v2275_data * v454_bc));
              float v2333_data = ir7[9];
              ir7[9] = (v2333_data + (v2275_data * v460_bc));
              float v2339_data = ir7[10];
              ir7[10] = (v2339_data + (v2275_data * v466_bc));
              float v2345_data = ir7[11];
              ir7[11] = (v2345_data + (v2275_data * v472_bc));
              float v2347_data = r6[6];
              float v2351_data = ir7[0];
              ir7[0] = (v2351_data + (v2347_data * v478_bc));
              float v2357_data = ir7[1];
              ir7[1] = (v2357_data + (v2347_data * v484_bc));
              float v2363_data = ir7[2];
              ir7[2] = (v2363_data + (v2347_data * v490_bc));
              float v2369_data = ir7[3];
              ir7[3] = (v2369_data + (v2347_data * v496_bc));
              float v2375_data = ir7[4];
              ir7[4] = (v2375_data + (v2347_data * v502_bc));
              float v2381_data = ir7[5];
              ir7[5] = (v2381_data + (v2347_data * v508_bc));
              float v2387_data = ir7[6];
              ir7[6] = (v2387_data + (v2347_data * v514_bc));
              float v2393_data = ir7[7];
              ir7[7] = (v2393_data + (v2347_data * v520_bc));
              float v2399_data = ir7[8];
              ir7[8] = (v2399_data + (v2347_data * v526_bc));
              float v2405_data = ir7[9];
              ir7[9] = (v2405_data + (v2347_data * v532_bc));
              float v2411_data = ir7[10];
              ir7[10] = (v2411_data + (v2347_data * v538_bc));
              float v2417_data = ir7[11];
              ir7[11] = (v2417_data + (v2347_data * v544_bc));
              float v2419_data = r6[7];
              float v2423_data = ir7[0];
              ir7[0] = (v2423_data + (v2419_data * v550_bc));
              float v2429_data = ir7[1];
              ir7[1] = (v2429_data + (v2419_data * v556_bc));
              float v2435_data = ir7[2];
              ir7[2] = (v2435_data + (v2419_data * v562_bc));
              float v2441_data = ir7[3];
              ir7[3] = (v2441_data + (v2419_data * v568_bc));
              float v2447_data = ir7[4];
              ir7[4] = (v2447_data + (v2419_data * v574_bc));
              float v2453_data = ir7[5];
              ir7[5] = (v2453_data + (v2419_data * v580_bc));
              float v2459_data = ir7[6];
              ir7[6] = (v2459_data + (v2419_data * v586_bc));
              float v2465_data = ir7[7];
              ir7[7] = (v2465_data + (v2419_data * v592_bc));
              float v2471_data = ir7[8];
              ir7[8] = (v2471_data + (v2419_data * v598_bc));
              float v2477_data = ir7[9];
              ir7[9] = (v2477_data + (v2419_data * v604_bc));
              float v2483_data = ir7[10];
              ir7[10] = (v2483_data + (v2419_data * v610_bc));
              float v2489_data = ir7[11];
              ir7[11] = (v2489_data + (v2419_data * v616_bc));
              float v2491_data = r6[8];
              float v2495_data = ir7[0];
              ir7[0] = (v2495_data + (v2491_data * v622_bc));
              float v2501_data = ir7[1];
              ir7[1] = (v2501_data + (v2491_data * v628_bc));
              float v2507_data = ir7[2];
              ir7[2] = (v2507_data + (v2491_data * v634_bc));
              float v2513_data = ir7[3];
              ir7[3] = (v2513_data + (v2491_data * v640_bc));
              float v2519_data = ir7[4];
              ir7[4] = (v2519_data + (v2491_data * v646_bc));
              float v2525_data = ir7[5];
              ir7[5] = (v2525_data + (v2491_data * v652_bc));
              float v2531_data = ir7[6];
              ir7[6] = (v2531_data + (v2491_data * v658_bc));
              float v2537_data = ir7[7];
              ir7[7] = (v2537_data + (v2491_data * v664_bc));
              float v2543_data = ir7[8];
              ir7[8] = (v2543_data + (v2491_data * v670_bc));
              float v2549_data = ir7[9];
              ir7[9] = (v2549_data + (v2491_data * v676_bc));
              float v2555_data = ir7[10];
              ir7[10] = (v2555_data + (v2491_data * v682_bc));
              float v2561_data = ir7[11];
              ir7[11] = (v2561_data + (v2491_data * v688_bc));
              float v2563_data = r6[9];
              float v2567_data = ir7[0];
              ir7[0] = (v2567_data + (v2563_data * v694_bc));
              float v2573_data = ir7[1];
              ir7[1] = (v2573_data + (v2563_data * v700_bc));
              float v2579_data = ir7[2];
              ir7[2] = (v2579_data + (v2563_data * v706_bc));
              float v2585_data = ir7[3];
              ir7[3] = (v2585_data + (v2563_data * v712_bc));
              float v2591_data = ir7[4];
              ir7[4] = (v2591_data + (v2563_data * v718_bc));
              float v2597_data = ir7[5];
              ir7[5] = (v2597_data + (v2563_data * v724_bc));
              float v2603_data = ir7[6];
              ir7[6] = (v2603_data + (v2563_data * v730_bc));
              float v2609_data = ir7[7];
              ir7[7] = (v2609_data + (v2563_data * v736_bc));
              float v2615_data = ir7[8];
              ir7[8] = (v2615_data + (v2563_data * v742_bc));
              float v2621_data = ir7[9];
              ir7[9] = (v2621_data + (v2563_data * v748_bc));
              float v2627_data = ir7[10];
              ir7[10] = (v2627_data + (v2563_data * v754_bc));
              float v2633_data = ir7[11];
              ir7[11] = (v2633_data + (v2563_data * v760_bc));
              float v2635_data = r6[10];
              float v2639_data = ir7[0];
              ir7[0] = (v2639_data + (v2635_data * v766_bc));
              float v2645_data = ir7[1];
              ir7[1] = (v2645_data + (v2635_data * v772_bc));
              float v2651_data = ir7[2];
              ir7[2] = (v2651_data + (v2635_data * v778_bc));
              float v2657_data = ir7[3];
              ir7[3] = (v2657_data + (v2635_data * v784_bc));
              float v2663_data = ir7[4];
              ir7[4] = (v2663_data + (v2635_data * v790_bc));
              float v2669_data = ir7[5];
              ir7[5] = (v2669_data + (v2635_data * v796_bc));
              float v2675_data = ir7[6];
              ir7[6] = (v2675_data + (v2635_data * v802_bc));
              float v2681_data = ir7[7];
              ir7[7] = (v2681_data + (v2635_data * v808_bc));
              float v2687_data = ir7[8];
              ir7[8] = (v2687_data + (v2635_data * v814_bc));
              float v2693_data = ir7[9];
              ir7[9] = (v2693_data + (v2635_data * v820_bc));
              float v2699_data = ir7[10];
              ir7[10] = (v2699_data + (v2635_data * v826_bc));
              float v2705_data = ir7[11];
              ir7[11] = (v2705_data + (v2635_data * v832_bc));
              float v2707_data = r6[11];
              float v2711_data = ir7[0];
              ir7[0] = (v2711_data + (v2707_data * v838_bc));
              float v2717_data = ir7[1];
              ir7[1] = (v2717_data + (v2707_data * v844_bc));
              float v2723_data = ir7[2];
              ir7[2] = (v2723_data + (v2707_data * v850_bc));
              float v2729_data = ir7[3];
              ir7[3] = (v2729_data + (v2707_data * v856_bc));
              float v2735_data = ir7[4];
              ir7[4] = (v2735_data + (v2707_data * v862_bc));
              float v2741_data = ir7[5];
              ir7[5] = (v2741_data + (v2707_data * v868_bc));
              float v2747_data = ir7[6];
              ir7[6] = (v2747_data + (v2707_data * v874_bc));
              float v2753_data = ir7[7];
              ir7[7] = (v2753_data + (v2707_data * v880_bc));
              float v2759_data = ir7[8];
              ir7[8] = (v2759_data + (v2707_data * v886_bc));
              float v2765_data = ir7[9];
              ir7[9] = (v2765_data + (v2707_data * v892_bc));
              float v2771_data = ir7[10];
              ir7[10] = (v2771_data + (v2707_data * v898_bc));
              float v2777_data = ir7[11];
              ir7[11] = (v2777_data + (v2707_data * v904_bc));
              // r7 = ir7
              if (v1905_g) {
                #pragma unroll
                for (int32_t v2779_n1 = 0; v2779_n1 < 12; ++v2779_n1) {
                  float v2781_data = ir7[v2779_n1];
                  r7[v2779_n1] = v2781_data;
                }
              }
              // s0 = store{r>s, clear}(localShrMem0, r7);
              bool v2783_g = (v25_lead >= 8) && v35_g;
              sycl::group_barrier(item.get_sub_group());
              if (v2783_g) {
                #pragma unroll
                for (int32_t v2784_z1 = 0; v2784_z1 < 12; ++v2784_z1) {
                  int32_t v2789_a = v25_lead + (v2784_z1 * 12);
                  s0[(v2789_a ^ ((v2789_a >> 4) & 15))] = 0.0f;
                }
              }
              if (v1905_g) {
                int32_t v2798_off = v25_lead + 6;
                #pragma unroll
                for (int32_t v2793_i1 = 0; v2793_i1 < 12; ++v2793_i1) {
                  float v2795_data = r7[v2793_i1];
                  int32_t v2800_a = v2798_off + (v2793_i1 * 12);
                  s0[(v2800_a ^ ((v2800_a >> 4) & 15))] = v2795_data;
                }
              }
              float r8[12]{};
              // ir8 = +(s0)
              // [(0, 12), (0, 12)] []
              float ir8[12]{};
              int32_t v2811_sw = v25_lead ^ v1812_sw;
              sycl::group_barrier(item.get_sub_group());
              float v2812_data_pre = s0[v35_g ? (v2811_sw) : (0)];
              float v2812_data = v35_g ? (v2812_data_pre) : (0.0f);
              float v2813_data = ir8[0];
              ir8[0] = (v2813_data + v2812_data);
              float v2819_data_pre = s0[v35_g ? ((v1817_a ^ (v1818_sw & 15))) : (0)];
              float v2819_data = v35_g ? (v2819_data_pre) : (0.0f);
              float v2820_data = ir8[1];
              ir8[1] = (v2820_data + v2819_data);
              float v2826_data_pre = s0[v35_g ? ((v1824_a ^ (v1825_sw & 15))) : (0)];
              float v2826_data = v35_g ? (v2826_data_pre) : (0.0f);
              float v2827_data = ir8[2];
              ir8[2] = (v2827_data + v2826_data);
              float v2833_data_pre = s0[v35_g ? ((v1831_a ^ (v1832_sw & 15))) : (0)];
              float v2833_data = v35_g ? (v2833_data_pre) : (0.0f);
              float v2834_data = ir8[3];
              ir8[3] = (v2834_data + v2833_data);
              float v2840_data_pre = s0[v35_g ? ((v1838_a ^ (v1839_sw & 15))) : (0)];
              float v2840_data = v35_g ? (v2840_data_pre) : (0.0f);
              float v2841_data = ir8[4];
              ir8[4] = (v2841_data + v2840_data);
              float v2847_data_pre = s0[v35_g ? ((v1845_a ^ (v1846_sw & 15))) : (0)];
              float v2847_data = v35_g ? (v2847_data_pre) : (0.0f);
              float v2848_data = ir8[5];
              ir8[5] = (v2848_data + v2847_data);
              float v2854_data_pre = s0[v35_g ? ((v1852_a ^ (v1853_sw & 15))) : (0)];
              float v2854_data = v35_g ? (v2854_data_pre) : (0.0f);
              float v2855_data = ir8[6];
              ir8[6] = (v2855_data + v2854_data);
              float v2861_data_pre = s0[v35_g ? ((v1859_a ^ (v1860_sw & 15))) : (0)];
              float v2861_data = v35_g ? (v2861_data_pre) : (0.0f);
              float v2862_data = ir8[7];
              ir8[7] = (v2862_data + v2861_data);
              float v2868_data_pre = s0[v35_g ? ((v1866_a ^ (v1867_sw & 15))) : (0)];
              float v2868_data = v35_g ? (v2868_data_pre) : (0.0f);
              float v2869_data = ir8[8];
              ir8[8] = (v2869_data + v2868_data);
              float v2875_data_pre = s0[v35_g ? ((v1873_a ^ (v1874_sw & 15))) : (0)];
              float v2875_data = v35_g ? (v2875_data_pre) : (0.0f);
              float v2876_data = ir8[9];
              ir8[9] = (v2876_data + v2875_data);
              float v2882_data_pre = s0[v35_g ? ((v1880_a ^ (v1881_sw & 15))) : (0)];
              float v2882_data = v35_g ? (v2882_data_pre) : (0.0f);
              float v2883_data = ir8[10];
              ir8[10] = (v2883_data + v2882_data);
              float v2889_data_pre = s0[v35_g ? ((v1887_a ^ (v1888_sw & 15))) : (0)];
              float v2889_data = v35_g ? (v2889_data_pre) : (0.0f);
              float v2890_data = ir8[11];
              ir8[11] = (v2890_data + v2889_data);
              // r8 = ir8
              if (v35_g) {
                #pragma unroll
                for (int32_t v2892_n1 = 0; v2892_n1 < 12; ++v2892_n1) {
                  float v2894_data = ir8[v2892_n1];
                  r8[v2892_n1] = v2894_data;
                }
              }
              // glb_m5 = store{r>g}(r8);
              if (v35_g) {
                #pragma unroll
                for (int32_t v2895_i1 = 0; v2895_i1 < 12; ++v2895_i1) {
                  float v2897_data = r8[v2895_i1];
                  glb_m5[(v25_lead + (v2895_i1 * 12))] = v2897_data;
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

