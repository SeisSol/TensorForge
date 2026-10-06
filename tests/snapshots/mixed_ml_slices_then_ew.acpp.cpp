// === base name ===
kernel_43fe0bc3f206ee9f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_43fe0bc3f206ee9f = {{8, 2, 1}, 8, 8, 1, 2, 576, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_43fe0bc3f206ee9f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_43fe0bc3f206ee9f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_43fe0bc3f206ee9f(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 144 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_43fe0bc3f206ee9f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_43fe0bc3f206ee9f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_43fe0bc3f206ee9f(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_43fe0bc3f206ee9f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (144, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 2 per block = block 8x2x1, 576 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×4(8×4) {0..8}×{0..4} strided
        //   m2 8×4(8×4) {0..8}×{0..4} strided
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j]@{0..8}×{0..4} = m0[i,k] × m1[k,j]
        //   t0[i,j]@{0..8}×{4..8} = m0[i,k] × m2[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":144}],"shared_bytes":576,"shared_elements":144,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,4]],"name":"m1","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[72 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float * __restrict__ s0 = &localShrMem0[0];
          size_t v11_batchIdLane0 = item.get_local_id(1) % 2;
          int32_t v29_lead = item.get_local_id(2) % 8;
          for (size_t v12_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v12_batchIdGroup0 < numElements0; v12_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v13_row = v12_batchIdGroup0 + v11_batchIdLane0;
            const bool batchIdActive0 = v13_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v13_row]));
            size_t v15_batchId0 = batchIdActive0 ? v13_row : v12_batchIdGroup0;
            size_t v16_ahead1 = v15_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v18_batchId1 = (v16_ahead1 < numElements0) ? v16_ahead1 : v15_batchId0;
            const float *const __restrict__ glb_m0 = &m0[v15_batchId0 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v15_batchId0 * 32 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v15_batchId0 * 32 + 0 + m2_extraOffset];
            float *const __restrict__ glb_m3 = &m3[v15_batchId0 * 64 + 0 + m3_extraOffset];
            float r0[8]{};
            // r0 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v30_i0 = 0; v30_i0 < 1; ++v30_i0) {
              int32_t v33_lead = v29_lead + (v30_i0 * 8);
              #pragma unroll
              for (int32_t v31_i1 = 0; v31_i1 < 8; ++v31_i1) {
                float v36_data = glb_m0[(v33_lead + (v31_i1 * 8))];
                r0[(v30_i0 + v31_i1)] = v36_data;
              }
            }
            float r1[4]{};
            // r1 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v39_i0 = 0; v39_i0 < 1; ++v39_i0) {
              int32_t v42_lead = v29_lead + (v39_i0 * 8);
              #pragma unroll
              for (int32_t v40_i1 = 0; v40_i1 < 4; ++v40_i1) {
                float v45_data = glb_m1[(v42_lead + (v40_i1 * 8))];
                r1[(v39_i0 + v40_i1)] = v45_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m0););
            float r3[4]{};
            // r3 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v48_i0 = 0; v48_i0 < 1; ++v48_i0) {
              int32_t v51_lead = v29_lead + (v48_i0 * 8);
              #pragma unroll
              for (int32_t v49_i1 = 0; v49_i1 < 4; ++v49_i1) {
                float v54_data = glb_m2[(v51_lead + (v49_i1 * 8))];
                r3[(v48_i0 + v49_i1)] = v54_data;
              }
            }
            // wait(r1 = load{g>r}(glb_m1););
            float r2[4]{};
            // r2 = +(r0 * r1) + None
            // [(0, 8), (0, 4)] [(0, 8)]
            float v57_data = r0[0];
            float v58_data = r1[0];
            float v61_data = r2[0];
            r2[0] = (v61_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v64_data = r1[1];
            float v67_data = r2[1];
            r2[1] = (v67_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v70_data = r1[2];
            float v73_data = r2[2];
            r2[2] = (v73_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v76_data = r1[3];
            float v79_data = r2[3];
            r2[3] = (v79_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v81_data = r0[1];
            float v85_data = r2[0];
            r2[0] = (v85_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v91_data = r2[1];
            r2[1] = (v91_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v97_data = r2[2];
            r2[2] = (v97_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v103_data = r2[3];
            r2[3] = (v103_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v105_data = r0[2];
            float v109_data = r2[0];
            r2[0] = (v109_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v115_data = r2[1];
            r2[1] = (v115_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v121_data = r2[2];
            r2[2] = (v121_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v127_data = r2[3];
            r2[3] = (v127_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v129_data = r0[3];
            float v133_data = r2[0];
            r2[0] = (v133_data + (v129_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v139_data = r2[1];
            r2[1] = (v139_data + (v129_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v145_data = r2[2];
            r2[2] = (v145_data + (v129_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v151_data = r2[3];
            r2[3] = (v151_data + (v129_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v153_data = r0[4];
            float v157_data = r2[0];
            r2[0] = (v157_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v163_data = r2[1];
            r2[1] = (v163_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v169_data = r2[2];
            r2[2] = (v169_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v175_data = r2[3];
            r2[3] = (v175_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v177_data = r0[5];
            float v181_data = r2[0];
            r2[0] = (v181_data + (v177_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v187_data = r2[1];
            r2[1] = (v187_data + (v177_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v193_data = r2[2];
            r2[2] = (v193_data + (v177_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v199_data = r2[3];
            r2[3] = (v199_data + (v177_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v201_data = r0[6];
            float v205_data = r2[0];
            r2[0] = (v205_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v211_data = r2[1];
            r2[1] = (v211_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v217_data = r2[2];
            r2[2] = (v217_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v223_data = r2[3];
            r2[3] = (v223_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v225_data = r0[7];
            float v229_data = r2[0];
            r2[0] = (v229_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v235_data = r2[1];
            r2[1] = (v235_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v241_data = r2[2];
            r2[2] = (v241_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v247_data = r2[3];
            r2[3] = (v247_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // s0 = store{r>s}(localShrMem0, r2);
            #pragma unroll
            for (int32_t v249_i0 = 0; v249_i0 < 1; ++v249_i0) {
              int32_t v254_lead = v29_lead + (v249_i0 * 8);
              #pragma unroll
              for (int32_t v250_i1 = 0; v250_i1 < 4; ++v250_i1) {
                float v252_data = r2[(v249_i0 + v250_i1)];
                int32_t v256_a = v254_lead + (v250_i1 * 8);
                s0[(v256_a ^ ((v256_a >> 5) & 31))] = v252_data;
              }
            }
            // wait(r3 = load{g>r}(glb_m2););
            float r4[4]{};
            // ir4 = +(r0 * r3)
            // [(0, 8), (0, 4)] [(0, 8)]
            float ir4[4]{};
            float v263_data = r3[0];
            float v266_data = ir4[0];
            ir4[0] = (v266_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v263_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v269_data = r3[1];
            float v272_data = ir4[1];
            ir4[1] = (v272_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v269_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v275_data = r3[2];
            float v278_data = ir4[2];
            ir4[2] = (v278_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v275_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v281_data = r3[3];
            float v284_data = ir4[3];
            ir4[3] = (v284_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v281_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v290_data = ir4[0];
            ir4[0] = (v290_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v263_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v296_data = ir4[1];
            ir4[1] = (v296_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v269_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v302_data = ir4[2];
            ir4[2] = (v302_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v275_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v308_data = ir4[3];
            ir4[3] = (v308_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v281_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v314_data = ir4[0];
            ir4[0] = (v314_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v263_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v320_data = ir4[1];
            ir4[1] = (v320_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v269_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v326_data = ir4[2];
            ir4[2] = (v326_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v275_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v332_data = ir4[3];
            ir4[3] = (v332_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v281_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v338_data = ir4[0];
            ir4[0] = (v338_data + (v129_data * (sycl::select_from_group(item.get_sub_group(), v263_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v344_data = ir4[1];
            ir4[1] = (v344_data + (v129_data * (sycl::select_from_group(item.get_sub_group(), v269_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v350_data = ir4[2];
            ir4[2] = (v350_data + (v129_data * (sycl::select_from_group(item.get_sub_group(), v275_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v356_data = ir4[3];
            ir4[3] = (v356_data + (v129_data * (sycl::select_from_group(item.get_sub_group(), v281_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v362_data = ir4[0];
            ir4[0] = (v362_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v263_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v368_data = ir4[1];
            ir4[1] = (v368_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v269_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v374_data = ir4[2];
            ir4[2] = (v374_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v275_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v380_data = ir4[3];
            ir4[3] = (v380_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v281_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v386_data = ir4[0];
            ir4[0] = (v386_data + (v177_data * (sycl::select_from_group(item.get_sub_group(), v263_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v392_data = ir4[1];
            ir4[1] = (v392_data + (v177_data * (sycl::select_from_group(item.get_sub_group(), v269_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v398_data = ir4[2];
            ir4[2] = (v398_data + (v177_data * (sycl::select_from_group(item.get_sub_group(), v275_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v404_data = ir4[3];
            ir4[3] = (v404_data + (v177_data * (sycl::select_from_group(item.get_sub_group(), v281_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v410_data = ir4[0];
            ir4[0] = (v410_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v263_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v416_data = ir4[1];
            ir4[1] = (v416_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v269_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v422_data = ir4[2];
            ir4[2] = (v422_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v275_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v428_data = ir4[3];
            ir4[3] = (v428_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v281_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v434_data = ir4[0];
            ir4[0] = (v434_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v263_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v440_data = ir4[1];
            ir4[1] = (v440_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v269_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v446_data = ir4[2];
            ir4[2] = (v446_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v275_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v452_data = ir4[3];
            ir4[3] = (v452_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v281_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // r4 = ir4
            #pragma unroll
            for (int32_t v454_n0 = 0; v454_n0 < 1; ++v454_n0) {
              #pragma unroll
              for (int32_t v455_n1 = 0; v455_n1 < 4; ++v455_n1) {
                int32_t v456_a = v454_n0 + v455_n1;
                float v457_data = ir4[v456_a];
                r4[v456_a] = v457_data;
              }
            }
            // s0 = store{r>s}(localShrMem0, r4);
            #pragma unroll
            for (int32_t v458_i0 = 0; v458_i0 < 1; ++v458_i0) {
              int32_t v463_lead = v29_lead + (v458_i0 * 8);
              #pragma unroll
              for (int32_t v459_i1 = 0; v459_i1 < 4; ++v459_i1) {
                float v461_data = r4[(v458_i0 + v459_i1)];
                int32_t v466_a = v463_lead + ((v459_i1 + 4) * 8);
                s0[(v466_a ^ ((v466_a >> 5) & 31))] = v461_data;
              }
            }
            item.barrier();
            // glb_m3 = abs(s0)
            #pragma unroll
            for (int32_t v470_k0 = 0; v470_k0 < 1; ++v470_k0) {
              int32_t v473_lead = v29_lead + (v470_k0 * 8);
              #pragma unroll
              for (int32_t v471_k1 = 0; v471_k1 < 8; ++v471_k1) {
                int32_t v475_a = v473_lead + (v471_k1 * 8);
                float v479_data = s0[(v475_a ^ ((v475_a >> 5) & 31))];
                float v480_e = sycl::fabs(v479_data);
                if (batchIdActive0) {
                  glb_m3[v475_a] = v480_e;
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

