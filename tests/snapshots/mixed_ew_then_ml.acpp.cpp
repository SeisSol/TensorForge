// === base name ===
kernel_e3f9169d2813bf79

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e3f9169d2813bf79 = {{8, 2, 1}, 8, 8, 1, 2, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e3f9169d2813bf79(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e3f9169d2813bf79(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e3f9169d2813bf79(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (8, 2, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 2 - 1) / 2;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 8;
  config.block[1] = 2;
  config.block[2] = 1;
  config.sharedMemBytes = 16 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_e3f9169d2813bf79(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e3f9169d2813bf79(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e3f9169d2813bf79(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e3f9169d2813bf79(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (16, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 2 per block = block 8x2x1, 64 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   TMP = abs(A)
        //   m1[i,j] = t0[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":16}],"shared_bytes":64,"shared_elements":16,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[8 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          size_t v10_batchIdLane0 = item.get_local_id(1) % 2;
          int32_t v27_lead = item.get_local_id(2) % 8;
          for (size_t v11_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v11_batchIdGroup0 < numElements0; v11_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_row = v11_batchIdGroup0 + v10_batchIdLane0;
            const bool batchIdActive0 = v12_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v12_row]));
            size_t v14_batchId0 = batchIdActive0 ? v12_row : v11_batchIdGroup0;
            size_t v15_ahead1 = v14_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v17_batchId1 = (v15_ahead1 < numElements0) ? v15_ahead1 : v14_batchId0;
            const float *const __restrict__ glb_m0 = &m0[v14_batchId0 * 64 + 0 + m0_extraOffset];
            float *const __restrict__ glb_m1 = &m1[v14_batchId0 * 64 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v14_batchId0 * 64 + 0 + m2_extraOffset];
            float r1[8]{};
            // r1 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
              int32_t v31_lead = v27_lead + (v28_i0 * 8);
              #pragma unroll
              for (int32_t v29_i1 = 0; v29_i1 < 8; ++v29_i1) {
                float v34_data = glb_m2[(v31_lead + (v29_i1 * 8))];
                r1[(v28_i0 + v29_i1)] = v34_data;
              }
            }
            float r0[8]{};
            // r0 = abs(glb_m0)
            #pragma unroll
            for (int32_t v37_k0 = 0; v37_k0 < 1; ++v37_k0) {
              int32_t v40_lead = v27_lead + (v37_k0 * 8);
              #pragma unroll
              for (int32_t v38_k1 = 0; v38_k1 < 8; ++v38_k1) {
                float v43_data = glb_m0[(v40_lead + (v38_k1 * 8))];
                r0[(v37_k0 + v38_k1)] = (sycl::fabs(v43_data));
              }
            }
            // wait(r1 = load{g>r}(glb_m2););
            float r2[8]{};
            // ir2 = +(r0 * r1)
            // [(0, 8), (0, 8)] [(0, 8)]
            float ir2[8]{};
            float v48_data = r0[0];
            float v49_data = r1[0];
            float v52_data = ir2[0];
            ir2[0] = (v52_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v55_data = r1[1];
            float v58_data = ir2[1];
            ir2[1] = (v58_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v61_data = r1[2];
            float v64_data = ir2[2];
            ir2[2] = (v64_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v67_data = r1[3];
            float v70_data = ir2[3];
            ir2[3] = (v70_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v73_data = r1[4];
            float v76_data = ir2[4];
            ir2[4] = (v76_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v79_data = r1[5];
            float v82_data = ir2[5];
            ir2[5] = (v82_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v85_data = r1[6];
            float v88_data = ir2[6];
            ir2[6] = (v88_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v91_data = r1[7];
            float v94_data = ir2[7];
            ir2[7] = (v94_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v96_data = r0[1];
            float v100_data = ir2[0];
            ir2[0] = (v100_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v106_data = ir2[1];
            ir2[1] = (v106_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v112_data = ir2[2];
            ir2[2] = (v112_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v118_data = ir2[3];
            ir2[3] = (v118_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v124_data = ir2[4];
            ir2[4] = (v124_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v130_data = ir2[5];
            ir2[5] = (v130_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v136_data = ir2[6];
            ir2[6] = (v136_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v142_data = ir2[7];
            ir2[7] = (v142_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v144_data = r0[2];
            float v148_data = ir2[0];
            ir2[0] = (v148_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v154_data = ir2[1];
            ir2[1] = (v154_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v160_data = ir2[2];
            ir2[2] = (v160_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v166_data = ir2[3];
            ir2[3] = (v166_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v172_data = ir2[4];
            ir2[4] = (v172_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v178_data = ir2[5];
            ir2[5] = (v178_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v184_data = ir2[6];
            ir2[6] = (v184_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v190_data = ir2[7];
            ir2[7] = (v190_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v192_data = r0[3];
            float v196_data = ir2[0];
            ir2[0] = (v196_data + (v192_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v202_data = ir2[1];
            ir2[1] = (v202_data + (v192_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v208_data = ir2[2];
            ir2[2] = (v208_data + (v192_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v214_data = ir2[3];
            ir2[3] = (v214_data + (v192_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v220_data = ir2[4];
            ir2[4] = (v220_data + (v192_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v226_data = ir2[5];
            ir2[5] = (v226_data + (v192_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v232_data = ir2[6];
            ir2[6] = (v232_data + (v192_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v238_data = ir2[7];
            ir2[7] = (v238_data + (v192_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v240_data = r0[4];
            float v244_data = ir2[0];
            ir2[0] = (v244_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v250_data = ir2[1];
            ir2[1] = (v250_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v256_data = ir2[2];
            ir2[2] = (v256_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v262_data = ir2[3];
            ir2[3] = (v262_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v268_data = ir2[4];
            ir2[4] = (v268_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v274_data = ir2[5];
            ir2[5] = (v274_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v280_data = ir2[6];
            ir2[6] = (v280_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v286_data = ir2[7];
            ir2[7] = (v286_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v288_data = r0[5];
            float v292_data = ir2[0];
            ir2[0] = (v292_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v298_data = ir2[1];
            ir2[1] = (v298_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v304_data = ir2[2];
            ir2[2] = (v304_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v310_data = ir2[3];
            ir2[3] = (v310_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v316_data = ir2[4];
            ir2[4] = (v316_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v322_data = ir2[5];
            ir2[5] = (v322_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v328_data = ir2[6];
            ir2[6] = (v328_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v334_data = ir2[7];
            ir2[7] = (v334_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v336_data = r0[6];
            float v340_data = ir2[0];
            ir2[0] = (v340_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v346_data = ir2[1];
            ir2[1] = (v346_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v352_data = ir2[2];
            ir2[2] = (v352_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v358_data = ir2[3];
            ir2[3] = (v358_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v364_data = ir2[4];
            ir2[4] = (v364_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v370_data = ir2[5];
            ir2[5] = (v370_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v376_data = ir2[6];
            ir2[6] = (v376_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v382_data = ir2[7];
            ir2[7] = (v382_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v384_data = r0[7];
            float v388_data = ir2[0];
            ir2[0] = (v388_data + (v384_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v394_data = ir2[1];
            ir2[1] = (v394_data + (v384_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v400_data = ir2[2];
            ir2[2] = (v400_data + (v384_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v406_data = ir2[3];
            ir2[3] = (v406_data + (v384_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v412_data = ir2[4];
            ir2[4] = (v412_data + (v384_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v418_data = ir2[5];
            ir2[5] = (v418_data + (v384_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v424_data = ir2[6];
            ir2[6] = (v424_data + (v384_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v430_data = ir2[7];
            ir2[7] = (v430_data + (v384_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // r2 = ir2
            #pragma unroll
            for (int32_t v432_n0 = 0; v432_n0 < 1; ++v432_n0) {
              #pragma unroll
              for (int32_t v433_n1 = 0; v433_n1 < 8; ++v433_n1) {
                int32_t v434_a = v432_n0 + v433_n1;
                float v435_data = ir2[v434_a];
                r2[v434_a] = v435_data;
              }
            }
            // glb_m1 = store{r>g}(r2);
            #pragma unroll
            for (int32_t v436_i0 = 0; v436_i0 < 1; ++v436_i0) {
              #pragma unroll
              for (int32_t v437_i1 = 0; v437_i1 < 8; ++v437_i1) {
                float v439_data = r2[(v436_i0 + v437_i1)];
                if (batchIdActive0) {
                  glb_m1[((v27_lead + (v436_i0 * 8)) + (v437_i1 * 8))] = v439_data;
                }
              }
            }
            item.barrier();
          }
        }
      });
    }
  });
}

