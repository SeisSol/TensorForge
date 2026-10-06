// === base name ===
kernel_bbff68e62ab159e8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_bbff68e62ab159e8 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_bbff68e62ab159e8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_bbff68e62ab159e8(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_bbff68e62ab159e8(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_bbff68e62ab159e8(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_bbff68e62ab159e8(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_bbff68e62ab159e8(stream, grid, block, m0, m1, m1_extraOffset, m2, m2_extraOffset, m3, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_bbff68e62ab159e8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×20(16×17) {0..16}×{1..18} none
        //   m1 20×9(17×9) {1..18}×{0..9} strided
        //   m2 16×9(16×9) {0..16}×{0..9} strided
        //   m3 16×20(16×15) {0..16}×{1..16} none
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]@{1..16}×{0..9}
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"none","alias":"A1","bbox":[[0,1],[16,18]],"name":"m0","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,18]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"pointer_based","bbox":[[1,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const float *const __restrict__ glb_m0 = &m0[0];
          const float *const __restrict__ glb_m3 = &m3[0];
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 153 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 144 + 0 + m2_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v24_lead = item.get_local_id(2) % 16;
              if (v24_lead >= 1) {
                int32_t v29_a = v24_lead - 1;
                #pragma unroll
                for (int32_t v26_i1 = 0; v26_i1 < 9; ++v26_i1) {
                  float v32_data = glb_m1[(v29_a + (v26_i1 * 17))];
                  r0[(v26_i1 * 2)] = v32_data;
                }
              }
              if (v24_lead < 2) {
                int32_t v39_a = (v24_lead + 16_i32) - 1;
                #pragma unroll
                for (int32_t v36_i1 = 0; v36_i1 < 9; ++v36_i1) {
                  float v42_data = glb_m1[(v39_a + (v36_i1 * 17))];
                  r0[(1 + (v36_i1 * 2))] = v42_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r1[9]{};
              // r1 = +(glb_m0 * r0) + None
              // [(0, 16), (0, 9)] [(1, 18)]
              float v49_data = glb_m0[v24_lead];
              float v50_data = r0[0];
              float v53_data = r1[0];
              r1[0] = (v53_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v56_data = r0[2];
              float v59_data = r1[1];
              r1[1] = (v59_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v62_data = r0[4];
              float v65_data = r1[2];
              r1[2] = (v65_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v68_data = r0[6];
              float v71_data = r1[3];
              r1[3] = (v71_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v74_data = r0[8];
              float v77_data = r1[4];
              r1[4] = (v77_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v80_data = r0[10];
              float v83_data = r1[5];
              r1[5] = (v83_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v86_data = r0[12];
              float v89_data = r1[6];
              r1[6] = (v89_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v92_data = r0[14];
              float v95_data = r1[7];
              r1[7] = (v95_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v98_data = r0[16];
              float v101_data = r1[8];
              r1[8] = (v101_data + (v49_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              int32_t v103_a = v24_lead + 16;
              float v104_data = glb_m0[v103_a];
              float v108_data = r1[0];
              r1[0] = (v108_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v114_data = r1[1];
              r1[1] = (v114_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v120_data = r1[2];
              r1[2] = (v120_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v126_data = r1[3];
              r1[3] = (v126_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v132_data = r1[4];
              r1[4] = (v132_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v138_data = r1[5];
              r1[5] = (v138_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v144_data = r1[6];
              r1[6] = (v144_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v150_data = r1[7];
              r1[7] = (v150_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v156_data = r1[8];
              r1[8] = (v156_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              int32_t v158_a = v24_lead + 32;
              float v159_data = glb_m0[v158_a];
              float v163_data = r1[0];
              r1[0] = (v163_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v169_data = r1[1];
              r1[1] = (v169_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v175_data = r1[2];
              r1[2] = (v175_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v181_data = r1[3];
              r1[3] = (v181_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v187_data = r1[4];
              r1[4] = (v187_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v193_data = r1[5];
              r1[5] = (v193_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v199_data = r1[6];
              r1[6] = (v199_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v205_data = r1[7];
              r1[7] = (v205_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v211_data = r1[8];
              r1[8] = (v211_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              int32_t v213_a = v24_lead + 48;
              float v214_data = glb_m0[v213_a];
              float v218_data = r1[0];
              r1[0] = (v218_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v224_data = r1[1];
              r1[1] = (v224_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v230_data = r1[2];
              r1[2] = (v230_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v236_data = r1[3];
              r1[3] = (v236_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v242_data = r1[4];
              r1[4] = (v242_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v248_data = r1[5];
              r1[5] = (v248_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v254_data = r1[6];
              r1[6] = (v254_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v260_data = r1[7];
              r1[7] = (v260_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v266_data = r1[8];
              r1[8] = (v266_data + (v214_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              int32_t v268_a = v24_lead + 64;
              float v269_data = glb_m0[v268_a];
              float v273_data = r1[0];
              r1[0] = (v273_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v279_data = r1[1];
              r1[1] = (v279_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v285_data = r1[2];
              r1[2] = (v285_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v291_data = r1[3];
              r1[3] = (v291_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v297_data = r1[4];
              r1[4] = (v297_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v303_data = r1[5];
              r1[5] = (v303_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v309_data = r1[6];
              r1[6] = (v309_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v315_data = r1[7];
              r1[7] = (v315_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v321_data = r1[8];
              r1[8] = (v321_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              int32_t v323_a = v24_lead + 80;
              float v324_data = glb_m0[v323_a];
              float v328_data = r1[0];
              r1[0] = (v328_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v334_data = r1[1];
              r1[1] = (v334_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v340_data = r1[2];
              r1[2] = (v340_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v346_data = r1[3];
              r1[3] = (v346_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v352_data = r1[4];
              r1[4] = (v352_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v358_data = r1[5];
              r1[5] = (v358_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v364_data = r1[6];
              r1[6] = (v364_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v370_data = r1[7];
              r1[7] = (v370_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v376_data = r1[8];
              r1[8] = (v376_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              int32_t v378_a = v24_lead + 96;
              float v379_data = glb_m0[v378_a];
              float v383_data = r1[0];
              r1[0] = (v383_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v389_data = r1[1];
              r1[1] = (v389_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v395_data = r1[2];
              r1[2] = (v395_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v401_data = r1[3];
              r1[3] = (v401_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v407_data = r1[4];
              r1[4] = (v407_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v413_data = r1[5];
              r1[5] = (v413_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v419_data = r1[6];
              r1[6] = (v419_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v425_data = r1[7];
              r1[7] = (v425_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v431_data = r1[8];
              r1[8] = (v431_data + (v379_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              int32_t v433_a = v24_lead + 112;
              float v434_data = glb_m0[v433_a];
              float v438_data = r1[0];
              r1[0] = (v438_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v444_data = r1[1];
              r1[1] = (v444_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v450_data = r1[2];
              r1[2] = (v450_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v456_data = r1[3];
              r1[3] = (v456_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v462_data = r1[4];
              r1[4] = (v462_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v468_data = r1[5];
              r1[5] = (v468_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v474_data = r1[6];
              r1[6] = (v474_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v480_data = r1[7];
              r1[7] = (v480_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v486_data = r1[8];
              r1[8] = (v486_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              int32_t v488_a = v24_lead + 128;
              float v489_data = glb_m0[v488_a];
              float v493_data = r1[0];
              r1[0] = (v493_data + (v489_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v499_data = r1[1];
              r1[1] = (v499_data + (v489_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v505_data = r1[2];
              r1[2] = (v505_data + (v489_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v511_data = r1[3];
              r1[3] = (v511_data + (v489_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v517_data = r1[4];
              r1[4] = (v517_data + (v489_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v523_data = r1[5];
              r1[5] = (v523_data + (v489_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v529_data = r1[6];
              r1[6] = (v529_data + (v489_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v535_data = r1[7];
              r1[7] = (v535_data + (v489_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v541_data = r1[8];
              r1[8] = (v541_data + (v489_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              int32_t v543_a = v24_lead + 144;
              float v544_data = glb_m0[v543_a];
              float v548_data = r1[0];
              r1[0] = (v548_data + (v544_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v554_data = r1[1];
              r1[1] = (v554_data + (v544_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v560_data = r1[2];
              r1[2] = (v560_data + (v544_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v566_data = r1[3];
              r1[3] = (v566_data + (v544_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v572_data = r1[4];
              r1[4] = (v572_data + (v544_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v578_data = r1[5];
              r1[5] = (v578_data + (v544_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v584_data = r1[6];
              r1[6] = (v584_data + (v544_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v590_data = r1[7];
              r1[7] = (v590_data + (v544_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v596_data = r1[8];
              r1[8] = (v596_data + (v544_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              int32_t v598_a = v24_lead + 160;
              float v599_data = glb_m0[v598_a];
              float v603_data = r1[0];
              r1[0] = (v603_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v609_data = r1[1];
              r1[1] = (v609_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v615_data = r1[2];
              r1[2] = (v615_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v621_data = r1[3];
              r1[3] = (v621_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v627_data = r1[4];
              r1[4] = (v627_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v633_data = r1[5];
              r1[5] = (v633_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v639_data = r1[6];
              r1[6] = (v639_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v645_data = r1[7];
              r1[7] = (v645_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v651_data = r1[8];
              r1[8] = (v651_data + (v599_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              int32_t v653_a = v24_lead + 176;
              float v654_data = glb_m0[v653_a];
              float v658_data = r1[0];
              r1[0] = (v658_data + (v654_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v664_data = r1[1];
              r1[1] = (v664_data + (v654_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v670_data = r1[2];
              r1[2] = (v670_data + (v654_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v676_data = r1[3];
              r1[3] = (v676_data + (v654_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v682_data = r1[4];
              r1[4] = (v682_data + (v654_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v688_data = r1[5];
              r1[5] = (v688_data + (v654_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v694_data = r1[6];
              r1[6] = (v694_data + (v654_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v700_data = r1[7];
              r1[7] = (v700_data + (v654_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v706_data = r1[8];
              r1[8] = (v706_data + (v654_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              int32_t v708_a = v24_lead + 192;
              float v709_data = glb_m0[v708_a];
              float v713_data = r1[0];
              r1[0] = (v713_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v719_data = r1[1];
              r1[1] = (v719_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v725_data = r1[2];
              r1[2] = (v725_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v731_data = r1[3];
              r1[3] = (v731_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v737_data = r1[4];
              r1[4] = (v737_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v743_data = r1[5];
              r1[5] = (v743_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v749_data = r1[6];
              r1[6] = (v749_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v755_data = r1[7];
              r1[7] = (v755_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v761_data = r1[8];
              r1[8] = (v761_data + (v709_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              int32_t v763_a = v24_lead + 208;
              float v764_data = glb_m0[v763_a];
              float v768_data = r1[0];
              r1[0] = (v768_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v774_data = r1[1];
              r1[1] = (v774_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v780_data = r1[2];
              r1[2] = (v780_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v786_data = r1[3];
              r1[3] = (v786_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v792_data = r1[4];
              r1[4] = (v792_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v798_data = r1[5];
              r1[5] = (v798_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v804_data = r1[6];
              r1[6] = (v804_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v810_data = r1[7];
              r1[7] = (v810_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v816_data = r1[8];
              r1[8] = (v816_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              int32_t v818_a = v24_lead + 224;
              float v819_data = glb_m0[v818_a];
              float v823_data = r1[0];
              r1[0] = (v823_data + (v819_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v829_data = r1[1];
              r1[1] = (v829_data + (v819_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v835_data = r1[2];
              r1[2] = (v835_data + (v819_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v841_data = r1[3];
              r1[3] = (v841_data + (v819_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v847_data = r1[4];
              r1[4] = (v847_data + (v819_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v853_data = r1[5];
              r1[5] = (v853_data + (v819_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v859_data = r1[6];
              r1[6] = (v859_data + (v819_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v865_data = r1[7];
              r1[7] = (v865_data + (v819_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v871_data = r1[8];
              r1[8] = (v871_data + (v819_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v874_data = glb_m0[(v24_lead + 240)];
              float v875_data = r0[1];
              float v878_data = r1[0];
              r1[0] = (v878_data + (v874_data * (sycl::select_from_group(item.get_sub_group(), v875_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v881_data = r0[3];
              float v884_data = r1[1];
              r1[1] = (v884_data + (v874_data * (sycl::select_from_group(item.get_sub_group(), v881_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v887_data = r0[5];
              float v890_data = r1[2];
              r1[2] = (v890_data + (v874_data * (sycl::select_from_group(item.get_sub_group(), v887_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v893_data = r0[7];
              float v896_data = r1[3];
              r1[3] = (v896_data + (v874_data * (sycl::select_from_group(item.get_sub_group(), v893_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v899_data = r0[9];
              float v902_data = r1[4];
              r1[4] = (v902_data + (v874_data * (sycl::select_from_group(item.get_sub_group(), v899_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v905_data = r0[11];
              float v908_data = r1[5];
              r1[5] = (v908_data + (v874_data * (sycl::select_from_group(item.get_sub_group(), v905_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v911_data = r0[13];
              float v914_data = r1[6];
              r1[6] = (v914_data + (v874_data * (sycl::select_from_group(item.get_sub_group(), v911_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v917_data = r0[15];
              float v920_data = r1[7];
              r1[7] = (v920_data + (v874_data * (sycl::select_from_group(item.get_sub_group(), v917_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v923_data = r0[17];
              float v926_data = r1[8];
              r1[8] = (v926_data + (v874_data * (sycl::select_from_group(item.get_sub_group(), v923_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v929_data = glb_m0[(v24_lead + 256)];
              float v933_data = r1[0];
              r1[0] = (v933_data + (v929_data * (sycl::select_from_group(item.get_sub_group(), v875_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v939_data = r1[1];
              r1[1] = (v939_data + (v929_data * (sycl::select_from_group(item.get_sub_group(), v881_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v945_data = r1[2];
              r1[2] = (v945_data + (v929_data * (sycl::select_from_group(item.get_sub_group(), v887_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v951_data = r1[3];
              r1[3] = (v951_data + (v929_data * (sycl::select_from_group(item.get_sub_group(), v893_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v957_data = r1[4];
              r1[4] = (v957_data + (v929_data * (sycl::select_from_group(item.get_sub_group(), v899_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v963_data = r1[5];
              r1[5] = (v963_data + (v929_data * (sycl::select_from_group(item.get_sub_group(), v905_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v969_data = r1[6];
              r1[6] = (v969_data + (v929_data * (sycl::select_from_group(item.get_sub_group(), v911_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v975_data = r1[7];
              r1[7] = (v975_data + (v929_data * (sycl::select_from_group(item.get_sub_group(), v917_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v981_data = r1[8];
              r1[8] = (v981_data + (v929_data * (sycl::select_from_group(item.get_sub_group(), v923_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float r2[9]{};
              // ir2 = +(glb_m3 * r1)
              // [(0, 16), (0, 9)] [(1, 16)]
              float ir2[9]{};
              float v988_data = glb_m3[v24_lead];
              float v989_data = r1[0];
              float v992_data = ir2[0];
              ir2[0] = (v992_data + (v988_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v995_data = r1[1];
              float v998_data = ir2[1];
              ir2[1] = (v998_data + (v988_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1001_data = r1[2];
              float v1004_data = ir2[2];
              ir2[2] = (v1004_data + (v988_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1007_data = r1[3];
              float v1010_data = ir2[3];
              ir2[3] = (v1010_data + (v988_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1013_data = r1[4];
              float v1016_data = ir2[4];
              ir2[4] = (v1016_data + (v988_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1019_data = r1[5];
              float v1022_data = ir2[5];
              ir2[5] = (v1022_data + (v988_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1025_data = r1[6];
              float v1028_data = ir2[6];
              ir2[6] = (v1028_data + (v988_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1031_data = r1[7];
              float v1034_data = ir2[7];
              ir2[7] = (v1034_data + (v988_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1037_data = r1[8];
              float v1040_data = ir2[8];
              ir2[8] = (v1040_data + (v988_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1043_data = glb_m3[v103_a];
              float v1047_data = ir2[0];
              ir2[0] = (v1047_data + (v1043_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1053_data = ir2[1];
              ir2[1] = (v1053_data + (v1043_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1059_data = ir2[2];
              ir2[2] = (v1059_data + (v1043_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1065_data = ir2[3];
              ir2[3] = (v1065_data + (v1043_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1071_data = ir2[4];
              ir2[4] = (v1071_data + (v1043_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1077_data = ir2[5];
              ir2[5] = (v1077_data + (v1043_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1083_data = ir2[6];
              ir2[6] = (v1083_data + (v1043_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1089_data = ir2[7];
              ir2[7] = (v1089_data + (v1043_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1095_data = ir2[8];
              ir2[8] = (v1095_data + (v1043_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1098_data = glb_m3[v158_a];
              float v1102_data = ir2[0];
              ir2[0] = (v1102_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1108_data = ir2[1];
              ir2[1] = (v1108_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1114_data = ir2[2];
              ir2[2] = (v1114_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1120_data = ir2[3];
              ir2[3] = (v1120_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1126_data = ir2[4];
              ir2[4] = (v1126_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1132_data = ir2[5];
              ir2[5] = (v1132_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1138_data = ir2[6];
              ir2[6] = (v1138_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1144_data = ir2[7];
              ir2[7] = (v1144_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1150_data = ir2[8];
              ir2[8] = (v1150_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1153_data = glb_m3[v213_a];
              float v1157_data = ir2[0];
              ir2[0] = (v1157_data + (v1153_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1163_data = ir2[1];
              ir2[1] = (v1163_data + (v1153_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1169_data = ir2[2];
              ir2[2] = (v1169_data + (v1153_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1175_data = ir2[3];
              ir2[3] = (v1175_data + (v1153_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1181_data = ir2[4];
              ir2[4] = (v1181_data + (v1153_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1187_data = ir2[5];
              ir2[5] = (v1187_data + (v1153_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1193_data = ir2[6];
              ir2[6] = (v1193_data + (v1153_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1199_data = ir2[7];
              ir2[7] = (v1199_data + (v1153_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1205_data = ir2[8];
              ir2[8] = (v1205_data + (v1153_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1208_data = glb_m3[v268_a];
              float v1212_data = ir2[0];
              ir2[0] = (v1212_data + (v1208_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1218_data = ir2[1];
              ir2[1] = (v1218_data + (v1208_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1224_data = ir2[2];
              ir2[2] = (v1224_data + (v1208_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1230_data = ir2[3];
              ir2[3] = (v1230_data + (v1208_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1236_data = ir2[4];
              ir2[4] = (v1236_data + (v1208_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1242_data = ir2[5];
              ir2[5] = (v1242_data + (v1208_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1248_data = ir2[6];
              ir2[6] = (v1248_data + (v1208_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1254_data = ir2[7];
              ir2[7] = (v1254_data + (v1208_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1260_data = ir2[8];
              ir2[8] = (v1260_data + (v1208_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1263_data = glb_m3[v323_a];
              float v1267_data = ir2[0];
              ir2[0] = (v1267_data + (v1263_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1273_data = ir2[1];
              ir2[1] = (v1273_data + (v1263_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1279_data = ir2[2];
              ir2[2] = (v1279_data + (v1263_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1285_data = ir2[3];
              ir2[3] = (v1285_data + (v1263_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1291_data = ir2[4];
              ir2[4] = (v1291_data + (v1263_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1297_data = ir2[5];
              ir2[5] = (v1297_data + (v1263_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1303_data = ir2[6];
              ir2[6] = (v1303_data + (v1263_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1309_data = ir2[7];
              ir2[7] = (v1309_data + (v1263_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1315_data = ir2[8];
              ir2[8] = (v1315_data + (v1263_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1318_data = glb_m3[v378_a];
              float v1322_data = ir2[0];
              ir2[0] = (v1322_data + (v1318_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1328_data = ir2[1];
              ir2[1] = (v1328_data + (v1318_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1334_data = ir2[2];
              ir2[2] = (v1334_data + (v1318_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1340_data = ir2[3];
              ir2[3] = (v1340_data + (v1318_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1346_data = ir2[4];
              ir2[4] = (v1346_data + (v1318_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1352_data = ir2[5];
              ir2[5] = (v1352_data + (v1318_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1358_data = ir2[6];
              ir2[6] = (v1358_data + (v1318_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1364_data = ir2[7];
              ir2[7] = (v1364_data + (v1318_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1370_data = ir2[8];
              ir2[8] = (v1370_data + (v1318_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1373_data = glb_m3[v433_a];
              float v1377_data = ir2[0];
              ir2[0] = (v1377_data + (v1373_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1383_data = ir2[1];
              ir2[1] = (v1383_data + (v1373_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1389_data = ir2[2];
              ir2[2] = (v1389_data + (v1373_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1395_data = ir2[3];
              ir2[3] = (v1395_data + (v1373_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1401_data = ir2[4];
              ir2[4] = (v1401_data + (v1373_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1407_data = ir2[5];
              ir2[5] = (v1407_data + (v1373_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1413_data = ir2[6];
              ir2[6] = (v1413_data + (v1373_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1419_data = ir2[7];
              ir2[7] = (v1419_data + (v1373_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1425_data = ir2[8];
              ir2[8] = (v1425_data + (v1373_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1428_data = glb_m3[v488_a];
              float v1432_data = ir2[0];
              ir2[0] = (v1432_data + (v1428_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1438_data = ir2[1];
              ir2[1] = (v1438_data + (v1428_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1444_data = ir2[2];
              ir2[2] = (v1444_data + (v1428_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1450_data = ir2[3];
              ir2[3] = (v1450_data + (v1428_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1456_data = ir2[4];
              ir2[4] = (v1456_data + (v1428_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1462_data = ir2[5];
              ir2[5] = (v1462_data + (v1428_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1468_data = ir2[6];
              ir2[6] = (v1468_data + (v1428_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1474_data = ir2[7];
              ir2[7] = (v1474_data + (v1428_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1480_data = ir2[8];
              ir2[8] = (v1480_data + (v1428_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1483_data = glb_m3[v543_a];
              float v1487_data = ir2[0];
              ir2[0] = (v1487_data + (v1483_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1493_data = ir2[1];
              ir2[1] = (v1493_data + (v1483_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1499_data = ir2[2];
              ir2[2] = (v1499_data + (v1483_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1505_data = ir2[3];
              ir2[3] = (v1505_data + (v1483_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1511_data = ir2[4];
              ir2[4] = (v1511_data + (v1483_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1517_data = ir2[5];
              ir2[5] = (v1517_data + (v1483_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1523_data = ir2[6];
              ir2[6] = (v1523_data + (v1483_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1529_data = ir2[7];
              ir2[7] = (v1529_data + (v1483_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1535_data = ir2[8];
              ir2[8] = (v1535_data + (v1483_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1538_data = glb_m3[v598_a];
              float v1542_data = ir2[0];
              ir2[0] = (v1542_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1548_data = ir2[1];
              ir2[1] = (v1548_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1554_data = ir2[2];
              ir2[2] = (v1554_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1560_data = ir2[3];
              ir2[3] = (v1560_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1566_data = ir2[4];
              ir2[4] = (v1566_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1572_data = ir2[5];
              ir2[5] = (v1572_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1578_data = ir2[6];
              ir2[6] = (v1578_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1584_data = ir2[7];
              ir2[7] = (v1584_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1590_data = ir2[8];
              ir2[8] = (v1590_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1593_data = glb_m3[v653_a];
              float v1597_data = ir2[0];
              ir2[0] = (v1597_data + (v1593_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1603_data = ir2[1];
              ir2[1] = (v1603_data + (v1593_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1609_data = ir2[2];
              ir2[2] = (v1609_data + (v1593_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1615_data = ir2[3];
              ir2[3] = (v1615_data + (v1593_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1621_data = ir2[4];
              ir2[4] = (v1621_data + (v1593_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1627_data = ir2[5];
              ir2[5] = (v1627_data + (v1593_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1633_data = ir2[6];
              ir2[6] = (v1633_data + (v1593_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1639_data = ir2[7];
              ir2[7] = (v1639_data + (v1593_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1645_data = ir2[8];
              ir2[8] = (v1645_data + (v1593_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1648_data = glb_m3[v708_a];
              float v1652_data = ir2[0];
              ir2[0] = (v1652_data + (v1648_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1658_data = ir2[1];
              ir2[1] = (v1658_data + (v1648_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1664_data = ir2[2];
              ir2[2] = (v1664_data + (v1648_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1670_data = ir2[3];
              ir2[3] = (v1670_data + (v1648_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1676_data = ir2[4];
              ir2[4] = (v1676_data + (v1648_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1682_data = ir2[5];
              ir2[5] = (v1682_data + (v1648_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1688_data = ir2[6];
              ir2[6] = (v1688_data + (v1648_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1694_data = ir2[7];
              ir2[7] = (v1694_data + (v1648_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1700_data = ir2[8];
              ir2[8] = (v1700_data + (v1648_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1703_data = glb_m3[v763_a];
              float v1707_data = ir2[0];
              ir2[0] = (v1707_data + (v1703_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1713_data = ir2[1];
              ir2[1] = (v1713_data + (v1703_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1719_data = ir2[2];
              ir2[2] = (v1719_data + (v1703_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1725_data = ir2[3];
              ir2[3] = (v1725_data + (v1703_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1731_data = ir2[4];
              ir2[4] = (v1731_data + (v1703_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1737_data = ir2[5];
              ir2[5] = (v1737_data + (v1703_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1743_data = ir2[6];
              ir2[6] = (v1743_data + (v1703_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1749_data = ir2[7];
              ir2[7] = (v1749_data + (v1703_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1755_data = ir2[8];
              ir2[8] = (v1755_data + (v1703_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1758_data = glb_m3[v818_a];
              float v1762_data = ir2[0];
              ir2[0] = (v1762_data + (v1758_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1768_data = ir2[1];
              ir2[1] = (v1768_data + (v1758_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1774_data = ir2[2];
              ir2[2] = (v1774_data + (v1758_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1780_data = ir2[3];
              ir2[3] = (v1780_data + (v1758_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1786_data = ir2[4];
              ir2[4] = (v1786_data + (v1758_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1792_data = ir2[5];
              ir2[5] = (v1792_data + (v1758_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1798_data = ir2[6];
              ir2[6] = (v1798_data + (v1758_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1804_data = ir2[7];
              ir2[7] = (v1804_data + (v1758_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1810_data = ir2[8];
              ir2[8] = (v1810_data + (v1758_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              // r2 = ir2
              #pragma unroll
              for (int32_t v1812_n0 = 0; v1812_n0 < 1; ++v1812_n0) {
                #pragma unroll
                for (int32_t v1813_n1 = 0; v1813_n1 < 9; ++v1813_n1) {
                  int32_t v1814_a = v1812_n0 + v1813_n1;
                  float v1815_data = ir2[v1814_a];
                  r2[v1814_a] = v1815_data;
                }
              }
              // glb_m2 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1816_i0 = 0; v1816_i0 < 1; ++v1816_i0) {
                int32_t v1821_lead = v24_lead + (v1816_i0 * 16);
                #pragma unroll
                for (int32_t v1817_i1 = 0; v1817_i1 < 9; ++v1817_i1) {
                  float v1819_data = r2[(v1816_i0 + v1817_i1)];
                  glb_m2[(v1821_lead + (v1817_i1 * 16))] = v1819_data;
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

