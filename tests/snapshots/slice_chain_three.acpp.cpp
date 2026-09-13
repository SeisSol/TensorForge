// === base name ===
kernel_eb322d3387eb0caa

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_eb322d3387eb0caa = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_eb322d3387eb0caa(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_eb322d3387eb0caa(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_eb322d3387eb0caa(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_eb322d3387eb0caa(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_eb322d3387eb0caa(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_eb322d3387eb0caa(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_eb322d3387eb0caa(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×6) {0..12}×{0..6} strided
        //   m1 32×32(6×6) {0..6}×{0..6} strided
        //   m2 32×32(12×6) {0..12}×{0..6} strided
        //   m3 32×32(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,6]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[6,6]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,6]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[6,6]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
              const float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v3_batchId0 * 144 + 0 + m3_extraOffset];
              float r0[6]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v18_lead = item.get_local_id(2) % 16;
              bool v19_g = v18_lead < 12;
              if (v19_g) {
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 6; ++v20_i1) {
                  float v25_data = glb_m0[(v18_lead + (v20_i1 * 12))];
                  r0[v20_i1] = v25_data;
                }
              }
              float r1[6]{};
              // r1 = load{g>r}(glb_m1);
              if (v18_lead < 6) {
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 6; ++v29_i1) {
                  float v34_data = glb_m1[(v18_lead + (v29_i1 * 6))];
                  r1[v29_i1] = v34_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v19_g) {
                #pragma unroll
                for (int32_t v37_i1 = 0; v37_i1 < 12; ++v37_i1) {
                  float v42_data = glb_m3[(v18_lead + (v37_i1 * 12))];
                  r3[v37_i1] = v42_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[6]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              float v45_data = r0[0];
              float v46_data = r1[0];
              float v49_data = r2[0];
              r2[0] = (v49_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v52_data = r1[1];
              float v55_data = r2[1];
              r2[1] = (v55_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v58_data = r1[2];
              float v61_data = r2[2];
              r2[2] = (v61_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v64_data = r1[3];
              float v67_data = r2[3];
              r2[3] = (v67_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v70_data = r1[4];
              float v73_data = r2[4];
              r2[4] = (v73_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v76_data = r1[5];
              float v79_data = r2[5];
              r2[5] = (v79_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v81_data = r0[1];
              float v85_data = r2[0];
              r2[0] = (v85_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v91_data = r2[1];
              r2[1] = (v91_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v97_data = r2[2];
              r2[2] = (v97_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v103_data = r2[3];
              r2[3] = (v103_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v109_data = r2[4];
              r2[4] = (v109_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v115_data = r2[5];
              r2[5] = (v115_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v117_data = r0[2];
              float v121_data = r2[0];
              r2[0] = (v121_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v127_data = r2[1];
              r2[1] = (v127_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v133_data = r2[2];
              r2[2] = (v133_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v139_data = r2[3];
              r2[3] = (v139_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v145_data = r2[4];
              r2[4] = (v145_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v151_data = r2[5];
              r2[5] = (v151_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v153_data = r0[3];
              float v157_data = r2[0];
              r2[0] = (v157_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v163_data = r2[1];
              r2[1] = (v163_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v169_data = r2[2];
              r2[2] = (v169_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v175_data = r2[3];
              r2[3] = (v175_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v181_data = r2[4];
              r2[4] = (v181_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v187_data = r2[5];
              r2[5] = (v187_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v189_data = r0[4];
              float v193_data = r2[0];
              r2[0] = (v193_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v199_data = r2[1];
              r2[1] = (v199_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v205_data = r2[2];
              r2[2] = (v205_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v211_data = r2[3];
              r2[3] = (v211_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v217_data = r2[4];
              r2[4] = (v217_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v223_data = r2[5];
              r2[5] = (v223_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v225_data = r0[5];
              float v229_data = r2[0];
              r2[0] = (v229_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v235_data = r2[1];
              r2[1] = (v235_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v241_data = r2[2];
              r2[2] = (v241_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v247_data = r2[3];
              r2[3] = (v247_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v253_data = r2[4];
              r2[4] = (v253_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v259_data = r2[5];
              r2[5] = (v259_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              // wait(r3 = load{g>r}(glb_m3););
              float r4[6]{};
              // r4 = +(r3 * r2) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir4[6]{};
              float v263_data = r3[0];
              float v264_data = r2[0];
              float v267_data = ir4[0];
              ir4[0] = (v267_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v270_data = r2[1];
              float v273_data = ir4[1];
              ir4[1] = (v273_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v276_data = r2[2];
              float v279_data = ir4[2];
              ir4[2] = (v279_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v282_data = r2[3];
              float v285_data = ir4[3];
              ir4[3] = (v285_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v288_data = r2[4];
              float v291_data = ir4[4];
              ir4[4] = (v291_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v294_data = r2[5];
              float v297_data = ir4[5];
              ir4[5] = (v297_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v299_data = r3[1];
              float v303_data = ir4[0];
              ir4[0] = (v303_data + (v299_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v309_data = ir4[1];
              ir4[1] = (v309_data + (v299_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v315_data = ir4[2];
              ir4[2] = (v315_data + (v299_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v321_data = ir4[3];
              ir4[3] = (v321_data + (v299_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v327_data = ir4[4];
              ir4[4] = (v327_data + (v299_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v333_data = ir4[5];
              ir4[5] = (v333_data + (v299_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v335_data = r3[2];
              float v339_data = ir4[0];
              ir4[0] = (v339_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v345_data = ir4[1];
              ir4[1] = (v345_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v351_data = ir4[2];
              ir4[2] = (v351_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v357_data = ir4[3];
              ir4[3] = (v357_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v363_data = ir4[4];
              ir4[4] = (v363_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v369_data = ir4[5];
              ir4[5] = (v369_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v371_data = r3[3];
              float v375_data = ir4[0];
              ir4[0] = (v375_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v381_data = ir4[1];
              ir4[1] = (v381_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v387_data = ir4[2];
              ir4[2] = (v387_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v393_data = ir4[3];
              ir4[3] = (v393_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v399_data = ir4[4];
              ir4[4] = (v399_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v405_data = ir4[5];
              ir4[5] = (v405_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v407_data = r3[4];
              float v411_data = ir4[0];
              ir4[0] = (v411_data + (v407_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v417_data = ir4[1];
              ir4[1] = (v417_data + (v407_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v423_data = ir4[2];
              ir4[2] = (v423_data + (v407_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v429_data = ir4[3];
              ir4[3] = (v429_data + (v407_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v435_data = ir4[4];
              ir4[4] = (v435_data + (v407_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v441_data = ir4[5];
              ir4[5] = (v441_data + (v407_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v443_data = r3[5];
              float v447_data = ir4[0];
              ir4[0] = (v447_data + (v443_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v453_data = ir4[1];
              ir4[1] = (v453_data + (v443_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v459_data = ir4[2];
              ir4[2] = (v459_data + (v443_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v465_data = ir4[3];
              ir4[3] = (v465_data + (v443_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v471_data = ir4[4];
              ir4[4] = (v471_data + (v443_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v477_data = ir4[5];
              ir4[5] = (v477_data + (v443_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v479_data = r3[6];
              float v483_data = ir4[0];
              ir4[0] = (v483_data + (v479_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v489_data = ir4[1];
              ir4[1] = (v489_data + (v479_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v495_data = ir4[2];
              ir4[2] = (v495_data + (v479_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v501_data = ir4[3];
              ir4[3] = (v501_data + (v479_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v507_data = ir4[4];
              ir4[4] = (v507_data + (v479_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v513_data = ir4[5];
              ir4[5] = (v513_data + (v479_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v515_data = r3[7];
              float v519_data = ir4[0];
              ir4[0] = (v519_data + (v515_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v525_data = ir4[1];
              ir4[1] = (v525_data + (v515_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v531_data = ir4[2];
              ir4[2] = (v531_data + (v515_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v537_data = ir4[3];
              ir4[3] = (v537_data + (v515_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v543_data = ir4[4];
              ir4[4] = (v543_data + (v515_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v549_data = ir4[5];
              ir4[5] = (v549_data + (v515_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v551_data = r3[8];
              float v555_data = ir4[0];
              ir4[0] = (v555_data + (v551_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v561_data = ir4[1];
              ir4[1] = (v561_data + (v551_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v567_data = ir4[2];
              ir4[2] = (v567_data + (v551_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v573_data = ir4[3];
              ir4[3] = (v573_data + (v551_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v579_data = ir4[4];
              ir4[4] = (v579_data + (v551_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v585_data = ir4[5];
              ir4[5] = (v585_data + (v551_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v587_data = r3[9];
              float v591_data = ir4[0];
              ir4[0] = (v591_data + (v587_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v597_data = ir4[1];
              ir4[1] = (v597_data + (v587_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v603_data = ir4[2];
              ir4[2] = (v603_data + (v587_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v609_data = ir4[3];
              ir4[3] = (v609_data + (v587_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v615_data = ir4[4];
              ir4[4] = (v615_data + (v587_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v621_data = ir4[5];
              ir4[5] = (v621_data + (v587_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v623_data = r3[10];
              float v627_data = ir4[0];
              ir4[0] = (v627_data + (v623_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v633_data = ir4[1];
              ir4[1] = (v633_data + (v623_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v639_data = ir4[2];
              ir4[2] = (v639_data + (v623_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v645_data = ir4[3];
              ir4[3] = (v645_data + (v623_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v651_data = ir4[4];
              ir4[4] = (v651_data + (v623_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v657_data = ir4[5];
              ir4[5] = (v657_data + (v623_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v659_data = r3[11];
              float v663_data = ir4[0];
              ir4[0] = (v663_data + (v659_data * (sycl::select_from_group(item.get_sub_group(), v264_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v669_data = ir4[1];
              ir4[1] = (v669_data + (v659_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v675_data = ir4[2];
              ir4[2] = (v675_data + (v659_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v681_data = ir4[3];
              ir4[3] = (v681_data + (v659_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v687_data = ir4[4];
              ir4[4] = (v687_data + (v659_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v693_data = ir4[5];
              ir4[5] = (v693_data + (v659_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              if (v19_g) {
                #pragma unroll
                for (int32_t v695_n1 = 0; v695_n1 < 6; ++v695_n1) {
                  float v697_data = ir4[v695_n1];
                  r4[v695_n1] = v697_data;
                }
              }
              // glb_m2 = store{r>g}(r4);
              if (v19_g) {
                #pragma unroll
                for (int32_t v698_i1 = 0; v698_i1 < 6; ++v698_i1) {
                  float v700_data = r4[v698_i1];
                  glb_m2[(v18_lead + (v698_i1 * 12))] = v700_data;
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

