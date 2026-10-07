// === base name ===
kernel_18b19c34eb90f45e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_18b19c34eb90f45e = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_18b19c34eb90f45e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_18b19c34eb90f45e(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_18b19c34eb90f45e(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_18b19c34eb90f45e(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_18b19c34eb90f45e(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_18b19c34eb90f45e(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_18b19c34eb90f45e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 12×16(12×16) {0..12}×{0..16} strided
        //   m1 20×12(20×12) {0..20}×{0..12} strided
        //   m2 20×16(20×16) {0..20}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[k,i] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,16]],"name":"m0","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[20,12]],"name":"m1","ordered":false,"parts":1,"shape":[20,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[20,16]],"name":"m2","ordered":false,"parts":1,"shape":[20,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[20,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,12]},{"addressing":"strided","bbox":[[0,0],[20,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,16]}],"permute":[[1,0],[0,1]],"target":[[-1,0],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 320 + 0 + m2_extraOffset];
              float r0[20]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v22_lead = item.get_local_id(2) % 16;
              bool v23_g = v22_lead < 12;
              #pragma unroll
              for (int32_t v19_i0 = 0; v19_i0 < 20; ++v19_i0) {
                if (v23_g) {
                  float v28_data = glb_m1[(v19_i0 + (v22_lead * 20))];
                  r0[v19_i0] = v28_data;
                }
              }
              float r1[32]{};
              // r1 = load{g>r}(glb_m2);
              int32_t v33_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v34_i0 = 0; v34_i0 < 1; ++v34_i0) {
                int32_t v37_lead = v33_lead + (v34_i0 * 16);
                #pragma unroll
                for (int32_t v35_i1 = 0; v35_i1 < 16; ++v35_i1) {
                  float v40_data = glb_m2[(v37_lead + (v35_i1 * 20))];
                  r1[(v34_i0 + (v35_i1 * 2))] = v40_data;
                }
              }
              if (v33_lead < 4) {
                int32_t v46_lead = v33_lead + 16_i32;
                #pragma unroll
                for (int32_t v44_i1 = 0; v44_i1 < 16; ++v44_i1) {
                  float v49_data = glb_m2[(v46_lead + (v44_i1 * 20))];
                  r1[(1 + (v44_i1 * 2))] = v49_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // ir2 = +(r0 * r1)
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir2[16]{};
              float v54_data = r0[0];
              float v55_data = r1[0];
              float v58_data = ir2[0];
              ir2[0] = (v58_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v61_data = r1[2];
              float v64_data = ir2[1];
              ir2[1] = (v64_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v67_data = r1[4];
              float v70_data = ir2[2];
              ir2[2] = (v70_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v73_data = r1[6];
              float v76_data = ir2[3];
              ir2[3] = (v76_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v79_data = r1[8];
              float v82_data = ir2[4];
              ir2[4] = (v82_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v85_data = r1[10];
              float v88_data = ir2[5];
              ir2[5] = (v88_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v91_data = r1[12];
              float v94_data = ir2[6];
              ir2[6] = (v94_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v97_data = r1[14];
              float v100_data = ir2[7];
              ir2[7] = (v100_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v103_data = r1[16];
              float v106_data = ir2[8];
              ir2[8] = (v106_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v109_data = r1[18];
              float v112_data = ir2[9];
              ir2[9] = (v112_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v115_data = r1[20];
              float v118_data = ir2[10];
              ir2[10] = (v118_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v121_data = r1[22];
              float v124_data = ir2[11];
              ir2[11] = (v124_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v127_data = r1[24];
              float v130_data = ir2[12];
              ir2[12] = (v130_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v133_data = r1[26];
              float v136_data = ir2[13];
              ir2[13] = (v136_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v139_data = r1[28];
              float v142_data = ir2[14];
              ir2[14] = (v142_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v145_data = r1[30];
              float v148_data = ir2[15];
              ir2[15] = (v148_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v150_data = r0[1];
              float v154_data = ir2[0];
              ir2[0] = (v154_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v160_data = ir2[1];
              ir2[1] = (v160_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v166_data = ir2[2];
              ir2[2] = (v166_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v172_data = ir2[3];
              ir2[3] = (v172_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v178_data = ir2[4];
              ir2[4] = (v178_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v184_data = ir2[5];
              ir2[5] = (v184_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v190_data = ir2[6];
              ir2[6] = (v190_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v196_data = ir2[7];
              ir2[7] = (v196_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v202_data = ir2[8];
              ir2[8] = (v202_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v208_data = ir2[9];
              ir2[9] = (v208_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v214_data = ir2[10];
              ir2[10] = (v214_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v220_data = ir2[11];
              ir2[11] = (v220_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v226_data = ir2[12];
              ir2[12] = (v226_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v232_data = ir2[13];
              ir2[13] = (v232_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v238_data = ir2[14];
              ir2[14] = (v238_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v244_data = ir2[15];
              ir2[15] = (v244_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v246_data = r0[2];
              float v250_data = ir2[0];
              ir2[0] = (v250_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v256_data = ir2[1];
              ir2[1] = (v256_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v262_data = ir2[2];
              ir2[2] = (v262_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v268_data = ir2[3];
              ir2[3] = (v268_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v274_data = ir2[4];
              ir2[4] = (v274_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v280_data = ir2[5];
              ir2[5] = (v280_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v286_data = ir2[6];
              ir2[6] = (v286_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v292_data = ir2[7];
              ir2[7] = (v292_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v298_data = ir2[8];
              ir2[8] = (v298_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v304_data = ir2[9];
              ir2[9] = (v304_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v310_data = ir2[10];
              ir2[10] = (v310_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v316_data = ir2[11];
              ir2[11] = (v316_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v322_data = ir2[12];
              ir2[12] = (v322_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v328_data = ir2[13];
              ir2[13] = (v328_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v334_data = ir2[14];
              ir2[14] = (v334_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v340_data = ir2[15];
              ir2[15] = (v340_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v342_data = r0[3];
              float v346_data = ir2[0];
              ir2[0] = (v346_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v352_data = ir2[1];
              ir2[1] = (v352_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v358_data = ir2[2];
              ir2[2] = (v358_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v364_data = ir2[3];
              ir2[3] = (v364_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v370_data = ir2[4];
              ir2[4] = (v370_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v376_data = ir2[5];
              ir2[5] = (v376_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v382_data = ir2[6];
              ir2[6] = (v382_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v388_data = ir2[7];
              ir2[7] = (v388_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v394_data = ir2[8];
              ir2[8] = (v394_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v400_data = ir2[9];
              ir2[9] = (v400_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v406_data = ir2[10];
              ir2[10] = (v406_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v412_data = ir2[11];
              ir2[11] = (v412_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v418_data = ir2[12];
              ir2[12] = (v418_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v424_data = ir2[13];
              ir2[13] = (v424_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v430_data = ir2[14];
              ir2[14] = (v430_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v436_data = ir2[15];
              ir2[15] = (v436_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v438_data = r0[4];
              float v442_data = ir2[0];
              ir2[0] = (v442_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v448_data = ir2[1];
              ir2[1] = (v448_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v454_data = ir2[2];
              ir2[2] = (v454_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v460_data = ir2[3];
              ir2[3] = (v460_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v466_data = ir2[4];
              ir2[4] = (v466_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v472_data = ir2[5];
              ir2[5] = (v472_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v478_data = ir2[6];
              ir2[6] = (v478_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v484_data = ir2[7];
              ir2[7] = (v484_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v490_data = ir2[8];
              ir2[8] = (v490_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v496_data = ir2[9];
              ir2[9] = (v496_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v502_data = ir2[10];
              ir2[10] = (v502_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v508_data = ir2[11];
              ir2[11] = (v508_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v514_data = ir2[12];
              ir2[12] = (v514_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v520_data = ir2[13];
              ir2[13] = (v520_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v526_data = ir2[14];
              ir2[14] = (v526_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v532_data = ir2[15];
              ir2[15] = (v532_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v534_data = r0[5];
              float v538_data = ir2[0];
              ir2[0] = (v538_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v544_data = ir2[1];
              ir2[1] = (v544_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v550_data = ir2[2];
              ir2[2] = (v550_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v556_data = ir2[3];
              ir2[3] = (v556_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v562_data = ir2[4];
              ir2[4] = (v562_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v568_data = ir2[5];
              ir2[5] = (v568_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v574_data = ir2[6];
              ir2[6] = (v574_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v580_data = ir2[7];
              ir2[7] = (v580_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v586_data = ir2[8];
              ir2[8] = (v586_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v592_data = ir2[9];
              ir2[9] = (v592_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v598_data = ir2[10];
              ir2[10] = (v598_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v604_data = ir2[11];
              ir2[11] = (v604_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v610_data = ir2[12];
              ir2[12] = (v610_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v616_data = ir2[13];
              ir2[13] = (v616_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v622_data = ir2[14];
              ir2[14] = (v622_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v628_data = ir2[15];
              ir2[15] = (v628_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v630_data = r0[6];
              float v634_data = ir2[0];
              ir2[0] = (v634_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v640_data = ir2[1];
              ir2[1] = (v640_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v646_data = ir2[2];
              ir2[2] = (v646_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v652_data = ir2[3];
              ir2[3] = (v652_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v658_data = ir2[4];
              ir2[4] = (v658_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v664_data = ir2[5];
              ir2[5] = (v664_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v670_data = ir2[6];
              ir2[6] = (v670_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v676_data = ir2[7];
              ir2[7] = (v676_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v682_data = ir2[8];
              ir2[8] = (v682_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v688_data = ir2[9];
              ir2[9] = (v688_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v694_data = ir2[10];
              ir2[10] = (v694_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v700_data = ir2[11];
              ir2[11] = (v700_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v706_data = ir2[12];
              ir2[12] = (v706_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v712_data = ir2[13];
              ir2[13] = (v712_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v718_data = ir2[14];
              ir2[14] = (v718_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v724_data = ir2[15];
              ir2[15] = (v724_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v726_data = r0[7];
              float v730_data = ir2[0];
              ir2[0] = (v730_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v736_data = ir2[1];
              ir2[1] = (v736_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v742_data = ir2[2];
              ir2[2] = (v742_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v748_data = ir2[3];
              ir2[3] = (v748_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v754_data = ir2[4];
              ir2[4] = (v754_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v760_data = ir2[5];
              ir2[5] = (v760_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v766_data = ir2[6];
              ir2[6] = (v766_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v772_data = ir2[7];
              ir2[7] = (v772_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v778_data = ir2[8];
              ir2[8] = (v778_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v784_data = ir2[9];
              ir2[9] = (v784_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v790_data = ir2[10];
              ir2[10] = (v790_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v796_data = ir2[11];
              ir2[11] = (v796_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v802_data = ir2[12];
              ir2[12] = (v802_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v808_data = ir2[13];
              ir2[13] = (v808_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v814_data = ir2[14];
              ir2[14] = (v814_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v820_data = ir2[15];
              ir2[15] = (v820_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v822_data = r0[8];
              float v826_data = ir2[0];
              ir2[0] = (v826_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v832_data = ir2[1];
              ir2[1] = (v832_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v838_data = ir2[2];
              ir2[2] = (v838_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v844_data = ir2[3];
              ir2[3] = (v844_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v850_data = ir2[4];
              ir2[4] = (v850_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v856_data = ir2[5];
              ir2[5] = (v856_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v862_data = ir2[6];
              ir2[6] = (v862_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v868_data = ir2[7];
              ir2[7] = (v868_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v874_data = ir2[8];
              ir2[8] = (v874_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v880_data = ir2[9];
              ir2[9] = (v880_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v886_data = ir2[10];
              ir2[10] = (v886_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v892_data = ir2[11];
              ir2[11] = (v892_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v898_data = ir2[12];
              ir2[12] = (v898_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v904_data = ir2[13];
              ir2[13] = (v904_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v910_data = ir2[14];
              ir2[14] = (v910_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v916_data = ir2[15];
              ir2[15] = (v916_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v918_data = r0[9];
              float v922_data = ir2[0];
              ir2[0] = (v922_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v928_data = ir2[1];
              ir2[1] = (v928_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v934_data = ir2[2];
              ir2[2] = (v934_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v940_data = ir2[3];
              ir2[3] = (v940_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v946_data = ir2[4];
              ir2[4] = (v946_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v952_data = ir2[5];
              ir2[5] = (v952_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v958_data = ir2[6];
              ir2[6] = (v958_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v964_data = ir2[7];
              ir2[7] = (v964_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v970_data = ir2[8];
              ir2[8] = (v970_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v976_data = ir2[9];
              ir2[9] = (v976_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v982_data = ir2[10];
              ir2[10] = (v982_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v988_data = ir2[11];
              ir2[11] = (v988_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v994_data = ir2[12];
              ir2[12] = (v994_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1000_data = ir2[13];
              ir2[13] = (v1000_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1006_data = ir2[14];
              ir2[14] = (v1006_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1012_data = ir2[15];
              ir2[15] = (v1012_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1014_data = r0[10];
              float v1018_data = ir2[0];
              ir2[0] = (v1018_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1024_data = ir2[1];
              ir2[1] = (v1024_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1030_data = ir2[2];
              ir2[2] = (v1030_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1036_data = ir2[3];
              ir2[3] = (v1036_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1042_data = ir2[4];
              ir2[4] = (v1042_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1048_data = ir2[5];
              ir2[5] = (v1048_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1054_data = ir2[6];
              ir2[6] = (v1054_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1060_data = ir2[7];
              ir2[7] = (v1060_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1066_data = ir2[8];
              ir2[8] = (v1066_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1072_data = ir2[9];
              ir2[9] = (v1072_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1078_data = ir2[10];
              ir2[10] = (v1078_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1084_data = ir2[11];
              ir2[11] = (v1084_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1090_data = ir2[12];
              ir2[12] = (v1090_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1096_data = ir2[13];
              ir2[13] = (v1096_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1102_data = ir2[14];
              ir2[14] = (v1102_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1108_data = ir2[15];
              ir2[15] = (v1108_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1110_data = r0[11];
              float v1114_data = ir2[0];
              ir2[0] = (v1114_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1120_data = ir2[1];
              ir2[1] = (v1120_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1126_data = ir2[2];
              ir2[2] = (v1126_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1132_data = ir2[3];
              ir2[3] = (v1132_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1138_data = ir2[4];
              ir2[4] = (v1138_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1144_data = ir2[5];
              ir2[5] = (v1144_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1150_data = ir2[6];
              ir2[6] = (v1150_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1156_data = ir2[7];
              ir2[7] = (v1156_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1162_data = ir2[8];
              ir2[8] = (v1162_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1168_data = ir2[9];
              ir2[9] = (v1168_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1174_data = ir2[10];
              ir2[10] = (v1174_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1180_data = ir2[11];
              ir2[11] = (v1180_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1186_data = ir2[12];
              ir2[12] = (v1186_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1192_data = ir2[13];
              ir2[13] = (v1192_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1198_data = ir2[14];
              ir2[14] = (v1198_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1204_data = ir2[15];
              ir2[15] = (v1204_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1206_data = r0[12];
              float v1210_data = ir2[0];
              ir2[0] = (v1210_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1216_data = ir2[1];
              ir2[1] = (v1216_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1222_data = ir2[2];
              ir2[2] = (v1222_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1228_data = ir2[3];
              ir2[3] = (v1228_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1234_data = ir2[4];
              ir2[4] = (v1234_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1240_data = ir2[5];
              ir2[5] = (v1240_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1246_data = ir2[6];
              ir2[6] = (v1246_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1252_data = ir2[7];
              ir2[7] = (v1252_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1258_data = ir2[8];
              ir2[8] = (v1258_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1264_data = ir2[9];
              ir2[9] = (v1264_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1270_data = ir2[10];
              ir2[10] = (v1270_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1276_data = ir2[11];
              ir2[11] = (v1276_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1282_data = ir2[12];
              ir2[12] = (v1282_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1288_data = ir2[13];
              ir2[13] = (v1288_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1294_data = ir2[14];
              ir2[14] = (v1294_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1300_data = ir2[15];
              ir2[15] = (v1300_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1302_data = r0[13];
              float v1306_data = ir2[0];
              ir2[0] = (v1306_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1312_data = ir2[1];
              ir2[1] = (v1312_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1318_data = ir2[2];
              ir2[2] = (v1318_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1324_data = ir2[3];
              ir2[3] = (v1324_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1330_data = ir2[4];
              ir2[4] = (v1330_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1336_data = ir2[5];
              ir2[5] = (v1336_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1342_data = ir2[6];
              ir2[6] = (v1342_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1348_data = ir2[7];
              ir2[7] = (v1348_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1354_data = ir2[8];
              ir2[8] = (v1354_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1360_data = ir2[9];
              ir2[9] = (v1360_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1366_data = ir2[10];
              ir2[10] = (v1366_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1372_data = ir2[11];
              ir2[11] = (v1372_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1378_data = ir2[12];
              ir2[12] = (v1378_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1384_data = ir2[13];
              ir2[13] = (v1384_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1390_data = ir2[14];
              ir2[14] = (v1390_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1396_data = ir2[15];
              ir2[15] = (v1396_data + (v1302_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1398_data = r0[14];
              float v1402_data = ir2[0];
              ir2[0] = (v1402_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1408_data = ir2[1];
              ir2[1] = (v1408_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1414_data = ir2[2];
              ir2[2] = (v1414_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1420_data = ir2[3];
              ir2[3] = (v1420_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1426_data = ir2[4];
              ir2[4] = (v1426_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1432_data = ir2[5];
              ir2[5] = (v1432_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1438_data = ir2[6];
              ir2[6] = (v1438_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1444_data = ir2[7];
              ir2[7] = (v1444_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1450_data = ir2[8];
              ir2[8] = (v1450_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1456_data = ir2[9];
              ir2[9] = (v1456_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1462_data = ir2[10];
              ir2[10] = (v1462_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1468_data = ir2[11];
              ir2[11] = (v1468_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1474_data = ir2[12];
              ir2[12] = (v1474_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1480_data = ir2[13];
              ir2[13] = (v1480_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1486_data = ir2[14];
              ir2[14] = (v1486_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1492_data = ir2[15];
              ir2[15] = (v1492_data + (v1398_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1494_data = r0[15];
              float v1498_data = ir2[0];
              ir2[0] = (v1498_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1504_data = ir2[1];
              ir2[1] = (v1504_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1510_data = ir2[2];
              ir2[2] = (v1510_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1516_data = ir2[3];
              ir2[3] = (v1516_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1522_data = ir2[4];
              ir2[4] = (v1522_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1528_data = ir2[5];
              ir2[5] = (v1528_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1534_data = ir2[6];
              ir2[6] = (v1534_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1540_data = ir2[7];
              ir2[7] = (v1540_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1546_data = ir2[8];
              ir2[8] = (v1546_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1552_data = ir2[9];
              ir2[9] = (v1552_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1558_data = ir2[10];
              ir2[10] = (v1558_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1564_data = ir2[11];
              ir2[11] = (v1564_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1570_data = ir2[12];
              ir2[12] = (v1570_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1576_data = ir2[13];
              ir2[13] = (v1576_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1582_data = ir2[14];
              ir2[14] = (v1582_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1588_data = ir2[15];
              ir2[15] = (v1588_data + (v1494_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1590_data = r0[16];
              float v1591_data = r1[1];
              float v1594_data = ir2[0];
              ir2[0] = (v1594_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1591_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1597_data = r1[3];
              float v1600_data = ir2[1];
              ir2[1] = (v1600_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1597_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1603_data = r1[5];
              float v1606_data = ir2[2];
              ir2[2] = (v1606_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1609_data = r1[7];
              float v1612_data = ir2[3];
              ir2[3] = (v1612_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1609_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1615_data = r1[9];
              float v1618_data = ir2[4];
              ir2[4] = (v1618_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1615_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1621_data = r1[11];
              float v1624_data = ir2[5];
              ir2[5] = (v1624_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1621_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1627_data = r1[13];
              float v1630_data = ir2[6];
              ir2[6] = (v1630_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1627_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1633_data = r1[15];
              float v1636_data = ir2[7];
              ir2[7] = (v1636_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1633_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1639_data = r1[17];
              float v1642_data = ir2[8];
              ir2[8] = (v1642_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1639_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1645_data = r1[19];
              float v1648_data = ir2[9];
              ir2[9] = (v1648_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1645_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1651_data = r1[21];
              float v1654_data = ir2[10];
              ir2[10] = (v1654_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1651_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1657_data = r1[23];
              float v1660_data = ir2[11];
              ir2[11] = (v1660_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1657_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1663_data = r1[25];
              float v1666_data = ir2[12];
              ir2[12] = (v1666_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1663_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1669_data = r1[27];
              float v1672_data = ir2[13];
              ir2[13] = (v1672_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1669_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1675_data = r1[29];
              float v1678_data = ir2[14];
              ir2[14] = (v1678_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1675_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1681_data = r1[31];
              float v1684_data = ir2[15];
              ir2[15] = (v1684_data + (v1590_data * (sycl::select_from_group(item.get_sub_group(), v1681_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1686_data = r0[17];
              float v1690_data = ir2[0];
              ir2[0] = (v1690_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1591_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1696_data = ir2[1];
              ir2[1] = (v1696_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1597_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1702_data = ir2[2];
              ir2[2] = (v1702_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1708_data = ir2[3];
              ir2[3] = (v1708_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1609_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1714_data = ir2[4];
              ir2[4] = (v1714_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1615_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1720_data = ir2[5];
              ir2[5] = (v1720_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1621_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1726_data = ir2[6];
              ir2[6] = (v1726_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1627_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1732_data = ir2[7];
              ir2[7] = (v1732_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1633_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1738_data = ir2[8];
              ir2[8] = (v1738_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1639_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1744_data = ir2[9];
              ir2[9] = (v1744_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1645_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1750_data = ir2[10];
              ir2[10] = (v1750_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1651_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1756_data = ir2[11];
              ir2[11] = (v1756_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1657_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1762_data = ir2[12];
              ir2[12] = (v1762_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1663_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1768_data = ir2[13];
              ir2[13] = (v1768_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1669_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1774_data = ir2[14];
              ir2[14] = (v1774_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1675_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1780_data = ir2[15];
              ir2[15] = (v1780_data + (v1686_data * (sycl::select_from_group(item.get_sub_group(), v1681_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1782_data = r0[18];
              float v1786_data = ir2[0];
              ir2[0] = (v1786_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1591_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1792_data = ir2[1];
              ir2[1] = (v1792_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1597_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1798_data = ir2[2];
              ir2[2] = (v1798_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1804_data = ir2[3];
              ir2[3] = (v1804_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1609_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1810_data = ir2[4];
              ir2[4] = (v1810_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1615_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1816_data = ir2[5];
              ir2[5] = (v1816_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1621_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1822_data = ir2[6];
              ir2[6] = (v1822_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1627_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1828_data = ir2[7];
              ir2[7] = (v1828_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1633_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1834_data = ir2[8];
              ir2[8] = (v1834_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1639_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1840_data = ir2[9];
              ir2[9] = (v1840_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1645_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1846_data = ir2[10];
              ir2[10] = (v1846_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1651_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1852_data = ir2[11];
              ir2[11] = (v1852_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1657_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1858_data = ir2[12];
              ir2[12] = (v1858_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1663_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1864_data = ir2[13];
              ir2[13] = (v1864_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1669_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1870_data = ir2[14];
              ir2[14] = (v1870_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1675_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1876_data = ir2[15];
              ir2[15] = (v1876_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1681_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1878_data = r0[19];
              float v1882_data = ir2[0];
              ir2[0] = (v1882_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1591_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1888_data = ir2[1];
              ir2[1] = (v1888_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1597_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1894_data = ir2[2];
              ir2[2] = (v1894_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1900_data = ir2[3];
              ir2[3] = (v1900_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1609_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1906_data = ir2[4];
              ir2[4] = (v1906_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1615_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1912_data = ir2[5];
              ir2[5] = (v1912_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1621_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1918_data = ir2[6];
              ir2[6] = (v1918_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1627_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1924_data = ir2[7];
              ir2[7] = (v1924_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1633_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1930_data = ir2[8];
              ir2[8] = (v1930_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1639_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1936_data = ir2[9];
              ir2[9] = (v1936_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1645_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1942_data = ir2[10];
              ir2[10] = (v1942_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1651_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1948_data = ir2[11];
              ir2[11] = (v1948_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1657_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1954_data = ir2[12];
              ir2[12] = (v1954_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1663_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1960_data = ir2[13];
              ir2[13] = (v1960_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1669_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1966_data = ir2[14];
              ir2[14] = (v1966_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1675_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1972_data = ir2[15];
              ir2[15] = (v1972_data + (v1878_data * (sycl::select_from_group(item.get_sub_group(), v1681_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              // r2 = ir2
              bool v1974_g = v33_lead < 12;
              if (v1974_g) {
                #pragma unroll
                for (int32_t v1975_n1 = 0; v1975_n1 < 16; ++v1975_n1) {
                  float v1977_data = ir2[v1975_n1];
                  r2[v1975_n1] = v1977_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v1974_g) {
                #pragma unroll
                for (int32_t v1979_i1 = 0; v1979_i1 < 16; ++v1979_i1) {
                  float v1981_data = r2[v1979_i1];
                  glb_m0[(v33_lead + (v1979_i1 * 12))] = v1981_data;
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

