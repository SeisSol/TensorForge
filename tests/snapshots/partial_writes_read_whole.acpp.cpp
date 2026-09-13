// === base name ===
kernel_b828f9f3b928bc64

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b828f9f3b928bc64 = {{32, 1, 1}, 32, 32, 1, 1, 1152, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b828f9f3b928bc64(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b828f9f3b928bc64(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b828f9f3b928bc64(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (32, 1, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 1;
  config.block[2] = 1;
  config.sharedMemBytes = 288 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_b828f9f3b928bc64(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b828f9f3b928bc64(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b828f9f3b928bc64(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b828f9f3b928bc64(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (288, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 1 per block = block 32x1x1, 1152 B shared, occupancy grid
        // operands:
        //   m0 32×9(32×9) {0..32}×{0..9} pointer_based
        //   m1 16×9(16×9) {0..16}×{0..9} pointer_based
        //   m2 16×9(16×9) {0..16}×{0..9} pointer_based
        //   m3 32×9(32×9) {0..32}×{0..9} pointer_based
        //   m4 9×9(9×9) {0..9}×{0..9} pointer_based
        // operations:
        //   t0[i,j] = m0[i,j]
        //   t0[i,j] += m1[i,j]
        //   t0[i,j] += m2[i,j]
        //   m3[i,j] = t0[i,k] × m4[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":288}],"shared_bytes":1152,"shared_elements":288,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,9]],"name":"m0","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"F0","bbox":[[0,0],[16,9]],"name":"m1","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"F1","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,9]],"name":"m3","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"M","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[288 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[288];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v4_batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v4_batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v4_batchId0][0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v4_batchId0][0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v4_batchId0][0 + m4_extraOffset];
              float r0[9]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v20_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
                int32_t v24_lead = v20_lead + (v21_i0 * 32);
                #pragma unroll
                for (int32_t v22_i1 = 0; v22_i1 < 9; ++v22_i1) {
                  float v27_data = glb_m0[(v24_lead + (v22_i1 * 32))];
                  r0[(v21_i0 + v22_i1)] = v27_data;
                }
              }
              float r2[9]{};
              // r2 = load{g>r}(glb_m1);
              bool v30_g = v20_lead < 16;
              if (v30_g) {
                #pragma unroll
                for (int32_t v31_i1 = 0; v31_i1 < 9; ++v31_i1) {
                  float v36_data = glb_m1[(v20_lead + (v31_i1 * 16))];
                  r2[v31_i1] = v36_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[9]{};
              // r1 = +(r0) + None
              // [(0, 32), (0, 9)] []
              float v39_data = r0[0];
              float v40_data = r1[0];
              r1[0] = (v40_data + v39_data);
              float v42_data = r0[1];
              float v43_data = r1[1];
              r1[1] = (v43_data + v42_data);
              float v45_data = r0[2];
              float v46_data = r1[2];
              r1[2] = (v46_data + v45_data);
              float v48_data = r0[3];
              float v49_data = r1[3];
              r1[3] = (v49_data + v48_data);
              float v51_data = r0[4];
              float v52_data = r1[4];
              r1[4] = (v52_data + v51_data);
              float v54_data = r0[5];
              float v55_data = r1[5];
              r1[5] = (v55_data + v54_data);
              float v57_data = r0[6];
              float v58_data = r1[6];
              r1[6] = (v58_data + v57_data);
              float v60_data = r0[7];
              float v61_data = r1[7];
              r1[7] = (v61_data + v60_data);
              float v63_data = r0[8];
              float v64_data = r1[8];
              r1[8] = (v64_data + v63_data);
              // s0 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v66_i0 = 0; v66_i0 < 1; ++v66_i0) {
                int32_t v71_lead = v20_lead + (v66_i0 * 32);
                #pragma unroll
                for (int32_t v67_i1 = 0; v67_i1 < 9; ++v67_i1) {
                  float v69_data = r1[(v66_i0 + v67_i1)];
                  int32_t v73_a = v71_lead + (v67_i1 * 32);
                  s0[(v73_a ^ ((v73_a >> 5) & 31))] = v69_data;
                }
              }
              float r4[9]{};
              // r4 = load{g>r}(glb_m2);
              if (v30_g) {
                #pragma unroll
                for (int32_t v78_i1 = 0; v78_i1 < 9; ++v78_i1) {
                  float v83_data = glb_m2[(v20_lead + (v78_i1 * 16))];
                  r4[v78_i1] = v83_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m1););
              float r3[9]{};
              item.barrier();
              // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir3[9]{};
              float v87_data = r2[0];
              float v88_data = ir3[0];
              ir3[0] = (v88_data + v87_data);
              float v90_data = r2[1];
              float v91_data = ir3[1];
              ir3[1] = (v91_data + v90_data);
              float v93_data = r2[2];
              float v94_data = ir3[2];
              ir3[2] = (v94_data + v93_data);
              float v96_data = r2[3];
              float v97_data = ir3[3];
              ir3[3] = (v97_data + v96_data);
              float v99_data = r2[4];
              float v100_data = ir3[4];
              ir3[4] = (v100_data + v99_data);
              float v102_data = r2[5];
              float v103_data = ir3[5];
              ir3[5] = (v103_data + v102_data);
              float v105_data = r2[6];
              float v106_data = ir3[6];
              ir3[6] = (v106_data + v105_data);
              float v108_data = r2[7];
              float v109_data = ir3[7];
              ir3[7] = (v109_data + v108_data);
              float v111_data = r2[8];
              float v112_data = ir3[8];
              ir3[8] = (v112_data + v111_data);
              if (v30_g) {
                #pragma unroll
                for (int32_t v114_n1 = 0; v114_n1 < 9; ++v114_n1) {
                  float v116_data = ir3[v114_n1];
                  int32_t v120_a = v20_lead + (v114_n1 * 32);
                  float v124_data = s0[(v120_a ^ ((v120_a >> 5) & 31))];
                  r3[v114_n1] = (v124_data + v116_data);
                }
              }
              item.barrier();
              // s0 = store{r>s}(localShrMem0, r3);
              if (v30_g) {
                #pragma unroll
                for (int32_t v126_i1 = 0; v126_i1 < 9; ++v126_i1) {
                  float v128_data = r3[v126_i1];
                  int32_t v132_a = v20_lead + (v126_i1 * 32);
                  s0[(v132_a ^ ((v132_a >> 5) & 31))] = v128_data;
                }
              }
              float r6[9]{};
              // r6 = load{g>r}(glb_m4);
              if (v20_lead < 9) {
                #pragma unroll
                for (int32_t v138_i1 = 0; v138_i1 < 9; ++v138_i1) {
                  float v143_data = glb_m4[(v20_lead + (v138_i1 * 9))];
                  r6[v138_i1] = v143_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m2););
              float r5[9]{};
              item.barrier();
              // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir5[9]{};
              float v147_data = r4[0];
              float v148_data = ir5[0];
              ir5[0] = (v148_data + v147_data);
              float v150_data = r4[1];
              float v151_data = ir5[1];
              ir5[1] = (v151_data + v150_data);
              float v153_data = r4[2];
              float v154_data = ir5[2];
              ir5[2] = (v154_data + v153_data);
              float v156_data = r4[3];
              float v157_data = ir5[3];
              ir5[3] = (v157_data + v156_data);
              float v159_data = r4[4];
              float v160_data = ir5[4];
              ir5[4] = (v160_data + v159_data);
              float v162_data = r4[5];
              float v163_data = ir5[5];
              ir5[5] = (v163_data + v162_data);
              float v165_data = r4[6];
              float v166_data = ir5[6];
              ir5[6] = (v166_data + v165_data);
              float v168_data = r4[7];
              float v169_data = ir5[7];
              ir5[7] = (v169_data + v168_data);
              float v171_data = r4[8];
              float v172_data = ir5[8];
              ir5[8] = (v172_data + v171_data);
              if (v30_g) {
                #pragma unroll
                for (int32_t v174_n1 = 0; v174_n1 < 9; ++v174_n1) {
                  float v176_data = ir5[v174_n1];
                  int32_t v180_a = v20_lead + (v174_n1 * 32);
                  float v184_data = s0[(v180_a ^ ((v180_a >> 5) & 31))];
                  r5[v174_n1] = (v184_data + v176_data);
                }
              }
              item.barrier();
              // s0 = store{r>s}(localShrMem0, r5);
              if (v30_g) {
                #pragma unroll
                for (int32_t v186_i1 = 0; v186_i1 < 9; ++v186_i1) {
                  float v188_data = r5[v186_i1];
                  int32_t v192_a = v20_lead + (v186_i1 * 32);
                  s0[(v192_a ^ ((v192_a >> 5) & 31))] = v188_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m4););
              float r7[9]{};
              item.barrier();
              // r7 = +(s0 * r6) + None
              // [(0, 32), (0, 9)] [(0, 9)]
              float ir7[9]{};
              float v204_data = s0[(v20_lead ^ ((v20_lead >> 5) & 31))];
              float v205_data = r6[0];
              float v208_data = ir7[0];
              ir7[0] = (v208_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v211_data = r6[1];
              float v214_data = ir7[1];
              ir7[1] = (v214_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v211_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v217_data = r6[2];
              float v220_data = ir7[2];
              ir7[2] = (v220_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v217_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v223_data = r6[3];
              float v226_data = ir7[3];
              ir7[3] = (v226_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v223_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v229_data = r6[4];
              float v232_data = ir7[4];
              ir7[4] = (v232_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v229_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v235_data = r6[5];
              float v238_data = ir7[5];
              ir7[5] = (v238_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v235_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v241_data = r6[6];
              float v244_data = ir7[6];
              ir7[6] = (v244_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v247_data = r6[7];
              float v250_data = ir7[7];
              ir7[7] = (v250_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v253_data = r6[8];
              float v256_data = ir7[8];
              ir7[8] = (v256_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              int32_t v258_a = v20_lead + 32;
              float v262_data = s0[(v258_a ^ ((v258_a >> 5) & 31))];
              float v266_data = ir7[0];
              ir7[0] = (v266_data + (v262_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v272_data = ir7[1];
              ir7[1] = (v272_data + (v262_data * (sycl::select_from_group(item.get_sub_group(), v211_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v278_data = ir7[2];
              ir7[2] = (v278_data + (v262_data * (sycl::select_from_group(item.get_sub_group(), v217_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v284_data = ir7[3];
              ir7[3] = (v284_data + (v262_data * (sycl::select_from_group(item.get_sub_group(), v223_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v290_data = ir7[4];
              ir7[4] = (v290_data + (v262_data * (sycl::select_from_group(item.get_sub_group(), v229_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v296_data = ir7[5];
              ir7[5] = (v296_data + (v262_data * (sycl::select_from_group(item.get_sub_group(), v235_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v302_data = ir7[6];
              ir7[6] = (v302_data + (v262_data * (sycl::select_from_group(item.get_sub_group(), v241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v308_data = ir7[7];
              ir7[7] = (v308_data + (v262_data * (sycl::select_from_group(item.get_sub_group(), v247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v314_data = ir7[8];
              ir7[8] = (v314_data + (v262_data * (sycl::select_from_group(item.get_sub_group(), v253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              int32_t v316_a = v20_lead + 64;
              float v320_data = s0[(v316_a ^ ((v316_a >> 5) & 31))];
              float v324_data = ir7[0];
              ir7[0] = (v324_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v330_data = ir7[1];
              ir7[1] = (v330_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v211_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v336_data = ir7[2];
              ir7[2] = (v336_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v217_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v342_data = ir7[3];
              ir7[3] = (v342_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v223_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v348_data = ir7[4];
              ir7[4] = (v348_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v229_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v354_data = ir7[5];
              ir7[5] = (v354_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v235_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v360_data = ir7[6];
              ir7[6] = (v360_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v366_data = ir7[7];
              ir7[7] = (v366_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v372_data = ir7[8];
              ir7[8] = (v372_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              int32_t v374_a = v20_lead + 96;
              float v378_data = s0[(v374_a ^ ((v374_a >> 5) & 31))];
              float v382_data = ir7[0];
              ir7[0] = (v382_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v388_data = ir7[1];
              ir7[1] = (v388_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v211_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v394_data = ir7[2];
              ir7[2] = (v394_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v217_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v400_data = ir7[3];
              ir7[3] = (v400_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v223_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v406_data = ir7[4];
              ir7[4] = (v406_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v229_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v412_data = ir7[5];
              ir7[5] = (v412_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v235_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v418_data = ir7[6];
              ir7[6] = (v418_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v424_data = ir7[7];
              ir7[7] = (v424_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v430_data = ir7[8];
              ir7[8] = (v430_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              int32_t v432_a = v20_lead + 128;
              float v436_data = s0[(v432_a ^ ((v432_a >> 5) & 31))];
              float v440_data = ir7[0];
              ir7[0] = (v440_data + (v436_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v446_data = ir7[1];
              ir7[1] = (v446_data + (v436_data * (sycl::select_from_group(item.get_sub_group(), v211_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v452_data = ir7[2];
              ir7[2] = (v452_data + (v436_data * (sycl::select_from_group(item.get_sub_group(), v217_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v458_data = ir7[3];
              ir7[3] = (v458_data + (v436_data * (sycl::select_from_group(item.get_sub_group(), v223_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v464_data = ir7[4];
              ir7[4] = (v464_data + (v436_data * (sycl::select_from_group(item.get_sub_group(), v229_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v470_data = ir7[5];
              ir7[5] = (v470_data + (v436_data * (sycl::select_from_group(item.get_sub_group(), v235_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v476_data = ir7[6];
              ir7[6] = (v476_data + (v436_data * (sycl::select_from_group(item.get_sub_group(), v241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v482_data = ir7[7];
              ir7[7] = (v482_data + (v436_data * (sycl::select_from_group(item.get_sub_group(), v247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v488_data = ir7[8];
              ir7[8] = (v488_data + (v436_data * (sycl::select_from_group(item.get_sub_group(), v253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              int32_t v490_a = v20_lead + 160;
              float v494_data = s0[(v490_a ^ ((v490_a >> 5) & 31))];
              float v498_data = ir7[0];
              ir7[0] = (v498_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v504_data = ir7[1];
              ir7[1] = (v504_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v211_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v510_data = ir7[2];
              ir7[2] = (v510_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v217_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v516_data = ir7[3];
              ir7[3] = (v516_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v223_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v522_data = ir7[4];
              ir7[4] = (v522_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v229_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v528_data = ir7[5];
              ir7[5] = (v528_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v235_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v534_data = ir7[6];
              ir7[6] = (v534_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v540_data = ir7[7];
              ir7[7] = (v540_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v546_data = ir7[8];
              ir7[8] = (v546_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              int32_t v548_a = v20_lead + 192;
              float v552_data = s0[(v548_a ^ ((v548_a >> 5) & 31))];
              float v556_data = ir7[0];
              ir7[0] = (v556_data + (v552_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v562_data = ir7[1];
              ir7[1] = (v562_data + (v552_data * (sycl::select_from_group(item.get_sub_group(), v211_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v568_data = ir7[2];
              ir7[2] = (v568_data + (v552_data * (sycl::select_from_group(item.get_sub_group(), v217_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v574_data = ir7[3];
              ir7[3] = (v574_data + (v552_data * (sycl::select_from_group(item.get_sub_group(), v223_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v580_data = ir7[4];
              ir7[4] = (v580_data + (v552_data * (sycl::select_from_group(item.get_sub_group(), v229_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v586_data = ir7[5];
              ir7[5] = (v586_data + (v552_data * (sycl::select_from_group(item.get_sub_group(), v235_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v592_data = ir7[6];
              ir7[6] = (v592_data + (v552_data * (sycl::select_from_group(item.get_sub_group(), v241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v598_data = ir7[7];
              ir7[7] = (v598_data + (v552_data * (sycl::select_from_group(item.get_sub_group(), v247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v604_data = ir7[8];
              ir7[8] = (v604_data + (v552_data * (sycl::select_from_group(item.get_sub_group(), v253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              int32_t v606_a = v20_lead + 224;
              float v610_data = s0[(v606_a ^ ((v606_a >> 5) & 31))];
              float v614_data = ir7[0];
              ir7[0] = (v614_data + (v610_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v620_data = ir7[1];
              ir7[1] = (v620_data + (v610_data * (sycl::select_from_group(item.get_sub_group(), v211_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v626_data = ir7[2];
              ir7[2] = (v626_data + (v610_data * (sycl::select_from_group(item.get_sub_group(), v217_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v632_data = ir7[3];
              ir7[3] = (v632_data + (v610_data * (sycl::select_from_group(item.get_sub_group(), v223_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v638_data = ir7[4];
              ir7[4] = (v638_data + (v610_data * (sycl::select_from_group(item.get_sub_group(), v229_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v644_data = ir7[5];
              ir7[5] = (v644_data + (v610_data * (sycl::select_from_group(item.get_sub_group(), v235_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v650_data = ir7[6];
              ir7[6] = (v650_data + (v610_data * (sycl::select_from_group(item.get_sub_group(), v241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v656_data = ir7[7];
              ir7[7] = (v656_data + (v610_data * (sycl::select_from_group(item.get_sub_group(), v247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v662_data = ir7[8];
              ir7[8] = (v662_data + (v610_data * (sycl::select_from_group(item.get_sub_group(), v253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              int32_t v664_a = v20_lead + 256;
              float v668_data = s0[(v664_a ^ ((v664_a >> 5) & 31))];
              float v672_data = ir7[0];
              ir7[0] = (v672_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v678_data = ir7[1];
              ir7[1] = (v678_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v211_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v684_data = ir7[2];
              ir7[2] = (v684_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v217_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v690_data = ir7[3];
              ir7[3] = (v690_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v223_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v696_data = ir7[4];
              ir7[4] = (v696_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v229_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v702_data = ir7[5];
              ir7[5] = (v702_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v235_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v708_data = ir7[6];
              ir7[6] = (v708_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v714_data = ir7[7];
              ir7[7] = (v714_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v720_data = ir7[8];
              ir7[8] = (v720_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              #pragma unroll
              for (int32_t v722_n0 = 0; v722_n0 < 1; ++v722_n0) {
                #pragma unroll
                for (int32_t v723_n1 = 0; v723_n1 < 9; ++v723_n1) {
                  int32_t v724_a = v722_n0 + v723_n1;
                  float v725_data = ir7[v724_a];
                  r7[v724_a] = v725_data;
                }
              }
              // glb_m3 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v726_i0 = 0; v726_i0 < 1; ++v726_i0) {
                int32_t v731_lead = v20_lead + (v726_i0 * 32);
                #pragma unroll
                for (int32_t v727_i1 = 0; v727_i1 < 9; ++v727_i1) {
                  float v729_data = r7[(v726_i0 + v727_i1)];
                  glb_m3[(v731_lead + (v727_i1 * 32))] = v729_data;
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

