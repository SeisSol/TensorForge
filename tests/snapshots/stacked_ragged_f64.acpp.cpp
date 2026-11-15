// === base name ===
kernel_211fa6b39b4cb3bd

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_211fa6b39b4cb3bd = {{16, 16, 1}, 16, 48, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_211fa6b39b4cb3bd(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_211fa6b39b4cb3bd(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_211fa6b39b4cb3bd(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 256 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_211fa6b39b4cb3bd(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_211fa6b39b4cb3bd(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_211fa6b39b4cb3bd(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_211fa6b39b4cb3bd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: amd_matrix_k4=1,lanes_per_mult=16
        // launch: 16 lanes (48 active) x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
        // operands:
        //   m0 48×8(48×8) {0..48}×{0..8} strided
        //   m1 48×24(48×24) {0..48}×{0..24} none ordered
        //   m2 24×8(24×8) {0..24}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":48,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":2048,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[48,8]],"name":"m0","ordered":false,"parts":1,"shape":[48,8],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[48,24]],"name":"m1","ordered":true,"parts":1,"shape":[48,24],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[24,8]],"name":"m2","ordered":false,"parts":1,"shape":[24,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[48,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[48,8]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[48,24]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[48,24]},{"addressing":"strided","bbox":[[0,0],[24,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[24,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const double *const __restrict__ glb_m1 = &m1[0];
          double r0[16]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v4139_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v4139_flagSecondAhead < numElements0) ? v4139_flagSecondAhead : v3_batchId1)];
          const double *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 192 + 0 + m2_extraOffset];
          int32_t v4147_lead = item.get_local_id(2) % 16;
          int32_t v4149_lead = v4147_lead + 16_i32;
          bool v4150_g = v4147_lead < 8;
          // r0 = load{g>r}(peel_glb_m2);
          #pragma unroll
          for (int32_t v4172_i0 = 0; v4172_i0 < 1; ++v4172_i0) {
            int32_t v4174_lead = v4147_lead + (v4172_i0 * 16);
            #pragma unroll
            for (int32_t v4175_i1 = 0; v4175_i1 < 8; ++v4175_i1) {
              double v4178_data = peel_glb_m2[(v4174_lead + (v4175_i1 * 24))];
              r0[(v4172_i0 + (v4175_i1 * 2))] = v4178_data;
            }
          }
          #pragma unroll
          for (int32_t v4181_i1 = 0; v4181_i1 < 8; ++v4181_i1) {
            double v4184_data_pre = peel_glb_m2[v4150_g ? ((v4149_lead + (v4181_i1 * 24))) : (0)];
            double v4184_data = v4150_g ? (v4184_data_pre) : (0.0);
            r0[(1 + (v4181_i1 * 2))] = v4184_data;
          }
          uint32_t v4143_flagWord = flagWordFirst;
          uint32_t v4144_flagWordNext = flagWordSecond;
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const double *const __restrict__ wrap_glb_m2 = &m2[v11_batchId1 * 192 + 0 + m2_extraOffset];
            size_t v4204_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v4204_flagAheadAhead < numElements0) ? v4204_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v4143_flagWord);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v8_batchId0 * 384 + 0 + m0_extraOffset];
              double r1[24]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 48), (0, 8)] [(0, 24)]
              double ir1[24]{};
              int32_t v42_a = v4147_lead * 2;
              sycl::vec<double, 2> v44_data = *(sycl::vec<double, 2>*)&glb_m1[v42_a];
              double v45_p = v44_data[0];
              double v46_data = r0[0];
              double v47_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v53_data = r0[2];
              double v54_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v60_data = r0[4];
              double v61_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v67_data = r0[6];
              double v68_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v74_data = r0[8];
              double v75_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v81_data = r0[10];
              double v82_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v88_data = r0[12];
              double v89_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v95_data = r0[14];
              double v96_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v101_p = v44_data[1];
              int32_t v156_a = v42_a + 32;
              sycl::vec<double, 2> v158_data = *(sycl::vec<double, 2>*)&glb_m1[v156_a];
              double v159_p = v158_data[0];
              sycl::vec<double, 2> v215_data = *(sycl::vec<double, 2>*)&glb_m1[(64 + v42_a)];
              double v216_p = v215_data[0];
              double v218_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v225_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v232_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v239_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v246_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v253_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v260_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v267_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v272_p = v215_data[1];
              sycl::vec<double, 2> v328_data = *(sycl::vec<double, 2>*)&glb_m1[(64 + v156_a)];
              double v329_p = v328_data[0];
              sycl::vec<double, 2> v385_data = *(sycl::vec<double, 2>*)&glb_m1[(128 + v42_a)];
              double v386_p = v385_data[0];
              double v388_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v395_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v402_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v409_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v416_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v423_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v430_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v437_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v442_p = v385_data[1];
              sycl::vec<double, 2> v498_data = *(sycl::vec<double, 2>*)&glb_m1[(128 + v156_a)];
              double v499_p = v498_data[0];
              sycl::vec<double, 2> v555_data = *(sycl::vec<double, 2>*)&glb_m1[(192 + v42_a)];
              double v556_p = v555_data[0];
              double v558_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v565_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v572_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v579_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v586_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v593_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v600_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v607_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v612_p = v555_data[1];
              sycl::vec<double, 2> v668_data = *(sycl::vec<double, 2>*)&glb_m1[(192 + v156_a)];
              double v669_p = v668_data[0];
              sycl::vec<double, 2> v725_data = *(sycl::vec<double, 2>*)&glb_m1[(256 + v42_a)];
              double v726_p = v725_data[0];
              double v728_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v735_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v742_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v749_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v756_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v763_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v770_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v777_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v782_p = v725_data[1];
              sycl::vec<double, 2> v838_data = *(sycl::vec<double, 2>*)&glb_m1[(256 + v156_a)];
              double v839_p = v838_data[0];
              sycl::vec<double, 2> v895_data = *(sycl::vec<double, 2>*)&glb_m1[(320 + v42_a)];
              double v896_p = v895_data[0];
              double v898_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v905_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v912_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v919_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v926_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v933_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v940_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v947_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v952_p = v895_data[1];
              sycl::vec<double, 2> v1008_data = *(sycl::vec<double, 2>*)&glb_m1[(320 + v156_a)];
              double v1009_p = v1008_data[0];
              sycl::vec<double, 2> v1065_data = *(sycl::vec<double, 2>*)&glb_m1[(384 + v42_a)];
              double v1066_p = v1065_data[0];
              double v1068_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1075_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1082_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1089_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1096_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1103_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1110_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1117_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1122_p = v1065_data[1];
              sycl::vec<double, 2> v1178_data = *(sycl::vec<double, 2>*)&glb_m1[(384 + v156_a)];
              double v1179_p = v1178_data[0];
              sycl::vec<double, 2> v1235_data = *(sycl::vec<double, 2>*)&glb_m1[(448 + v42_a)];
              double v1236_p = v1235_data[0];
              double v1238_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v1245_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v1252_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v1259_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v1266_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v1273_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v1280_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v1287_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v1292_p = v1235_data[1];
              sycl::vec<double, 2> v1348_data = *(sycl::vec<double, 2>*)&glb_m1[(448 + v156_a)];
              double v1349_p = v1348_data[0];
              sycl::vec<double, 2> v1405_data = *(sycl::vec<double, 2>*)&glb_m1[(512 + v42_a)];
              double v1406_p = v1405_data[0];
              double v1408_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v1415_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v1422_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v1429_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v1436_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v1443_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v1450_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v1457_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v1462_p = v1405_data[1];
              sycl::vec<double, 2> v1518_data = *(sycl::vec<double, 2>*)&glb_m1[(512 + v156_a)];
              double v1519_p = v1518_data[0];
              sycl::vec<double, 2> v1575_data = *(sycl::vec<double, 2>*)&glb_m1[(576 + v42_a)];
              double v1576_p = v1575_data[0];
              double v1578_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v1585_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v1592_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v1599_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v1606_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v1613_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v1620_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v1627_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v1632_p = v1575_data[1];
              sycl::vec<double, 2> v1688_data = *(sycl::vec<double, 2>*)&glb_m1[(576 + v156_a)];
              double v1689_p = v1688_data[0];
              sycl::vec<double, 2> v1745_data = *(sycl::vec<double, 2>*)&glb_m1[(640 + v42_a)];
              double v1746_p = v1745_data[0];
              double v1748_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v1755_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v1762_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v1769_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v1776_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v1783_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v1790_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v1797_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v1802_p = v1745_data[1];
              sycl::vec<double, 2> v1858_data = *(sycl::vec<double, 2>*)&glb_m1[(640 + v156_a)];
              double v1859_p = v1858_data[0];
              sycl::vec<double, 2> v1915_data = *(sycl::vec<double, 2>*)&glb_m1[(704 + v42_a)];
              double v1916_p = v1915_data[0];
              double v1918_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v1925_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v1932_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v1939_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v1946_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v1953_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v1960_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v1967_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v1972_p = v1915_data[1];
              sycl::vec<double, 2> v2028_data = *(sycl::vec<double, 2>*)&glb_m1[(704 + v156_a)];
              double v2029_p = v2028_data[0];
              sycl::vec<double, 2> v2085_data = *(sycl::vec<double, 2>*)&glb_m1[(768 + v42_a)];
              double v2086_p = v2085_data[0];
              double v2088_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2095_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2102_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2109_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2116_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2123_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2130_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2137_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2142_p = v2085_data[1];
              sycl::vec<double, 2> v2198_data = *(sycl::vec<double, 2>*)&glb_m1[(768 + v156_a)];
              double v2199_p = v2198_data[0];
              sycl::vec<double, 2> v2255_data = *(sycl::vec<double, 2>*)&glb_m1[(832 + v42_a)];
              double v2256_p = v2255_data[0];
              double v2258_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v2265_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v2272_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v2279_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v2286_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v2293_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v2300_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v2307_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v2312_p = v2255_data[1];
              sycl::vec<double, 2> v2368_data = *(sycl::vec<double, 2>*)&glb_m1[(832 + v156_a)];
              double v2369_p = v2368_data[0];
              sycl::vec<double, 2> v2425_data = *(sycl::vec<double, 2>*)&glb_m1[(896 + v42_a)];
              double v2426_p = v2425_data[0];
              double v2428_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v2435_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v2442_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v2449_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v2456_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v2463_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v2470_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v2477_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v2482_p = v2425_data[1];
              sycl::vec<double, 2> v2538_data = *(sycl::vec<double, 2>*)&glb_m1[(896 + v156_a)];
              double v2539_p = v2538_data[0];
              sycl::vec<double, 2> v2595_data = *(sycl::vec<double, 2>*)&glb_m1[(960 + v42_a)];
              double v2596_p = v2595_data[0];
              double v2598_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v2605_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v2612_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v2619_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v2626_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v2633_bc = sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v2640_bc = sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v2647_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v2652_p = v2595_data[1];
              sycl::vec<double, 2> v2708_data = *(sycl::vec<double, 2>*)&glb_m1[(960 + v156_a)];
              double v2709_p = v2708_data[0];
              sycl::vec<double, 2> v2765_data = *(sycl::vec<double, 2>*)&glb_m1[(1024 + v42_a)];
              double v2766_p = v2765_data[0];
              double v2767_data = r0[1];
              double v2768_bc = sycl::select_from_group(item.get_sub_group(), v2767_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v2774_data = r0[3];
              double v2775_bc = sycl::select_from_group(item.get_sub_group(), v2774_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v2781_data = r0[5];
              double v2782_bc = sycl::select_from_group(item.get_sub_group(), v2781_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v2788_data = r0[7];
              double v2789_bc = sycl::select_from_group(item.get_sub_group(), v2788_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v2795_data = r0[9];
              double v2796_bc = sycl::select_from_group(item.get_sub_group(), v2795_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v2802_data = r0[11];
              double v2803_bc = sycl::select_from_group(item.get_sub_group(), v2802_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v2809_data = r0[13];
              double v2810_bc = sycl::select_from_group(item.get_sub_group(), v2809_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v2816_data = r0[15];
              double v2817_bc = sycl::select_from_group(item.get_sub_group(), v2816_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v2822_p = v2765_data[1];
              sycl::vec<double, 2> v2878_data = *(sycl::vec<double, 2>*)&glb_m1[(1024 + v156_a)];
              double v2879_p = v2878_data[0];
              sycl::vec<double, 2> v2935_data = *(sycl::vec<double, 2>*)&glb_m1[(1088 + v42_a)];
              double v2936_p = v2935_data[0];
              double v2938_bc = sycl::select_from_group(item.get_sub_group(), v2767_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v2945_bc = sycl::select_from_group(item.get_sub_group(), v2774_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v2952_bc = sycl::select_from_group(item.get_sub_group(), v2781_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v2959_bc = sycl::select_from_group(item.get_sub_group(), v2788_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v2966_bc = sycl::select_from_group(item.get_sub_group(), v2795_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v2973_bc = sycl::select_from_group(item.get_sub_group(), v2802_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v2980_bc = sycl::select_from_group(item.get_sub_group(), v2809_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v2987_bc = sycl::select_from_group(item.get_sub_group(), v2816_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v2992_p = v2935_data[1];
              sycl::vec<double, 2> v3048_data = *(sycl::vec<double, 2>*)&glb_m1[(1088 + v156_a)];
              double v3049_p = v3048_data[0];
              sycl::vec<double, 2> v3105_data = *(sycl::vec<double, 2>*)&glb_m1[(1152 + v42_a)];
              double v3106_p = v3105_data[0];
              double v3108_bc = sycl::select_from_group(item.get_sub_group(), v2767_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3115_bc = sycl::select_from_group(item.get_sub_group(), v2774_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3122_bc = sycl::select_from_group(item.get_sub_group(), v2781_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3129_bc = sycl::select_from_group(item.get_sub_group(), v2788_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3136_bc = sycl::select_from_group(item.get_sub_group(), v2795_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3143_bc = sycl::select_from_group(item.get_sub_group(), v2802_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3150_bc = sycl::select_from_group(item.get_sub_group(), v2809_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3157_bc = sycl::select_from_group(item.get_sub_group(), v2816_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3162_p = v3105_data[1];
              sycl::vec<double, 2> v3218_data = *(sycl::vec<double, 2>*)&glb_m1[(1152 + v156_a)];
              double v3219_p = v3218_data[0];
              sycl::vec<double, 2> v3275_data = *(sycl::vec<double, 2>*)&glb_m1[(1216 + v42_a)];
              double v3276_p = v3275_data[0];
              double v3278_bc = sycl::select_from_group(item.get_sub_group(), v2767_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v3285_bc = sycl::select_from_group(item.get_sub_group(), v2774_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v3292_bc = sycl::select_from_group(item.get_sub_group(), v2781_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v3299_bc = sycl::select_from_group(item.get_sub_group(), v2788_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v3306_bc = sycl::select_from_group(item.get_sub_group(), v2795_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v3313_bc = sycl::select_from_group(item.get_sub_group(), v2802_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v3320_bc = sycl::select_from_group(item.get_sub_group(), v2809_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v3327_bc = sycl::select_from_group(item.get_sub_group(), v2816_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v3332_p = v3275_data[1];
              sycl::vec<double, 2> v3388_data = *(sycl::vec<double, 2>*)&glb_m1[(1216 + v156_a)];
              double v3389_p = v3388_data[0];
              sycl::vec<double, 2> v3445_data = *(sycl::vec<double, 2>*)&glb_m1[(1280 + v42_a)];
              double v3446_p = v3445_data[0];
              double v3448_bc = sycl::select_from_group(item.get_sub_group(), v2767_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v3455_bc = sycl::select_from_group(item.get_sub_group(), v2774_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v3462_bc = sycl::select_from_group(item.get_sub_group(), v2781_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v3469_bc = sycl::select_from_group(item.get_sub_group(), v2788_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v3476_bc = sycl::select_from_group(item.get_sub_group(), v2795_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v3483_bc = sycl::select_from_group(item.get_sub_group(), v2802_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v3490_bc = sycl::select_from_group(item.get_sub_group(), v2809_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v3497_bc = sycl::select_from_group(item.get_sub_group(), v2816_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v3502_p = v3445_data[1];
              sycl::vec<double, 2> v3558_data = *(sycl::vec<double, 2>*)&glb_m1[(1280 + v156_a)];
              double v3559_p = v3558_data[0];
              double v3578_p = ((((((((((((((((((((v159_p * v61_bc) + (v329_p * v232_bc)) + (v499_p * v402_bc)) + (v669_p * v572_bc)) + (v839_p * v742_bc)) + (v1009_p * v912_bc)) + (v1179_p * v1082_bc)) + (v1349_p * v1252_bc)) + (v1519_p * v1422_bc)) + (v1689_p * v1592_bc)) + (v1859_p * v1762_bc)) + (v2029_p * v1932_bc)) + (v2199_p * v2102_bc)) + (v2369_p * v2272_bc)) + (v2539_p * v2442_bc)) + (v2709_p * v2612_bc)) + (v2879_p * v2782_bc)) + (v3049_p * v2952_bc)) + (v3219_p * v3122_bc)) + (v3389_p * v3292_bc)) + (v3559_p * v3462_bc);
              double v3585_p = ((((((((((((((((((((v159_p * v68_bc) + (v329_p * v239_bc)) + (v499_p * v409_bc)) + (v669_p * v579_bc)) + (v839_p * v749_bc)) + (v1009_p * v919_bc)) + (v1179_p * v1089_bc)) + (v1349_p * v1259_bc)) + (v1519_p * v1429_bc)) + (v1689_p * v1599_bc)) + (v1859_p * v1769_bc)) + (v2029_p * v1939_bc)) + (v2199_p * v2109_bc)) + (v2369_p * v2279_bc)) + (v2539_p * v2449_bc)) + (v2709_p * v2619_bc)) + (v2879_p * v2789_bc)) + (v3049_p * v2959_bc)) + (v3219_p * v3129_bc)) + (v3389_p * v3299_bc)) + (v3559_p * v3469_bc);
              double v3592_p = ((((((((((((((((((((v159_p * v75_bc) + (v329_p * v246_bc)) + (v499_p * v416_bc)) + (v669_p * v586_bc)) + (v839_p * v756_bc)) + (v1009_p * v926_bc)) + (v1179_p * v1096_bc)) + (v1349_p * v1266_bc)) + (v1519_p * v1436_bc)) + (v1689_p * v1606_bc)) + (v1859_p * v1776_bc)) + (v2029_p * v1946_bc)) + (v2199_p * v2116_bc)) + (v2369_p * v2286_bc)) + (v2539_p * v2456_bc)) + (v2709_p * v2626_bc)) + (v2879_p * v2796_bc)) + (v3049_p * v2966_bc)) + (v3219_p * v3136_bc)) + (v3389_p * v3306_bc)) + (v3559_p * v3476_bc);
              double v3599_p = ((((((((((((((((((((v159_p * v82_bc) + (v329_p * v253_bc)) + (v499_p * v423_bc)) + (v669_p * v593_bc)) + (v839_p * v763_bc)) + (v1009_p * v933_bc)) + (v1179_p * v1103_bc)) + (v1349_p * v1273_bc)) + (v1519_p * v1443_bc)) + (v1689_p * v1613_bc)) + (v1859_p * v1783_bc)) + (v2029_p * v1953_bc)) + (v2199_p * v2123_bc)) + (v2369_p * v2293_bc)) + (v2539_p * v2463_bc)) + (v2709_p * v2633_bc)) + (v2879_p * v2803_bc)) + (v3049_p * v2973_bc)) + (v3219_p * v3143_bc)) + (v3389_p * v3313_bc)) + (v3559_p * v3483_bc);
              double v3606_p = ((((((((((((((((((((v159_p * v89_bc) + (v329_p * v260_bc)) + (v499_p * v430_bc)) + (v669_p * v600_bc)) + (v839_p * v770_bc)) + (v1009_p * v940_bc)) + (v1179_p * v1110_bc)) + (v1349_p * v1280_bc)) + (v1519_p * v1450_bc)) + (v1689_p * v1620_bc)) + (v1859_p * v1790_bc)) + (v2029_p * v1960_bc)) + (v2199_p * v2130_bc)) + (v2369_p * v2300_bc)) + (v2539_p * v2470_bc)) + (v2709_p * v2640_bc)) + (v2879_p * v2810_bc)) + (v3049_p * v2980_bc)) + (v3219_p * v3150_bc)) + (v3389_p * v3320_bc)) + (v3559_p * v3490_bc);
              double v3613_p = ((((((((((((((((((((v159_p * v96_bc) + (v329_p * v267_bc)) + (v499_p * v437_bc)) + (v669_p * v607_bc)) + (v839_p * v777_bc)) + (v1009_p * v947_bc)) + (v1179_p * v1117_bc)) + (v1349_p * v1287_bc)) + (v1519_p * v1457_bc)) + (v1689_p * v1627_bc)) + (v1859_p * v1797_bc)) + (v2029_p * v1967_bc)) + (v2199_p * v2137_bc)) + (v2369_p * v2307_bc)) + (v2539_p * v2477_bc)) + (v2709_p * v2647_bc)) + (v2879_p * v2817_bc)) + (v3049_p * v2987_bc)) + (v3219_p * v3157_bc)) + (v3389_p * v3327_bc)) + (v3559_p * v3497_bc);
              sycl::vec<double, 2> v3615_data = *(sycl::vec<double, 2>*)&glb_m1[(1344 + v42_a)];
              double v3616_p = v3615_data[0];
              double v3618_bc = sycl::select_from_group(item.get_sub_group(), v2767_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v3621_p = (((((((((((((((((((((v45_p * v47_bc) + (v216_p * v218_bc)) + (v386_p * v388_bc)) + (v556_p * v558_bc)) + (v726_p * v728_bc)) + (v896_p * v898_bc)) + (v1066_p * v1068_bc)) + (v1236_p * v1238_bc)) + (v1406_p * v1408_bc)) + (v1576_p * v1578_bc)) + (v1746_p * v1748_bc)) + (v1916_p * v1918_bc)) + (v2086_p * v2088_bc)) + (v2256_p * v2258_bc)) + (v2426_p * v2428_bc)) + (v2596_p * v2598_bc)) + (v2766_p * v2768_bc)) + (v2936_p * v2938_bc)) + (v3106_p * v3108_bc)) + (v3276_p * v3278_bc)) + (v3446_p * v3448_bc)) + (v3616_p * v3618_bc);
              double v3625_bc = sycl::select_from_group(item.get_sub_group(), v2774_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v3628_p = (((((((((((((((((((((v45_p * v54_bc) + (v216_p * v225_bc)) + (v386_p * v395_bc)) + (v556_p * v565_bc)) + (v726_p * v735_bc)) + (v896_p * v905_bc)) + (v1066_p * v1075_bc)) + (v1236_p * v1245_bc)) + (v1406_p * v1415_bc)) + (v1576_p * v1585_bc)) + (v1746_p * v1755_bc)) + (v1916_p * v1925_bc)) + (v2086_p * v2095_bc)) + (v2256_p * v2265_bc)) + (v2426_p * v2435_bc)) + (v2596_p * v2605_bc)) + (v2766_p * v2775_bc)) + (v2936_p * v2945_bc)) + (v3106_p * v3115_bc)) + (v3276_p * v3285_bc)) + (v3446_p * v3455_bc)) + (v3616_p * v3625_bc);
              double v3632_bc = sycl::select_from_group(item.get_sub_group(), v2781_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v3635_p = (((((((((((((((((((((v45_p * v61_bc) + (v216_p * v232_bc)) + (v386_p * v402_bc)) + (v556_p * v572_bc)) + (v726_p * v742_bc)) + (v896_p * v912_bc)) + (v1066_p * v1082_bc)) + (v1236_p * v1252_bc)) + (v1406_p * v1422_bc)) + (v1576_p * v1592_bc)) + (v1746_p * v1762_bc)) + (v1916_p * v1932_bc)) + (v2086_p * v2102_bc)) + (v2256_p * v2272_bc)) + (v2426_p * v2442_bc)) + (v2596_p * v2612_bc)) + (v2766_p * v2782_bc)) + (v2936_p * v2952_bc)) + (v3106_p * v3122_bc)) + (v3276_p * v3292_bc)) + (v3446_p * v3462_bc)) + (v3616_p * v3632_bc);
              double v3639_bc = sycl::select_from_group(item.get_sub_group(), v2788_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v3642_p = (((((((((((((((((((((v45_p * v68_bc) + (v216_p * v239_bc)) + (v386_p * v409_bc)) + (v556_p * v579_bc)) + (v726_p * v749_bc)) + (v896_p * v919_bc)) + (v1066_p * v1089_bc)) + (v1236_p * v1259_bc)) + (v1406_p * v1429_bc)) + (v1576_p * v1599_bc)) + (v1746_p * v1769_bc)) + (v1916_p * v1939_bc)) + (v2086_p * v2109_bc)) + (v2256_p * v2279_bc)) + (v2426_p * v2449_bc)) + (v2596_p * v2619_bc)) + (v2766_p * v2789_bc)) + (v2936_p * v2959_bc)) + (v3106_p * v3129_bc)) + (v3276_p * v3299_bc)) + (v3446_p * v3469_bc)) + (v3616_p * v3639_bc);
              double v3646_bc = sycl::select_from_group(item.get_sub_group(), v2795_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v3649_p = (((((((((((((((((((((v45_p * v75_bc) + (v216_p * v246_bc)) + (v386_p * v416_bc)) + (v556_p * v586_bc)) + (v726_p * v756_bc)) + (v896_p * v926_bc)) + (v1066_p * v1096_bc)) + (v1236_p * v1266_bc)) + (v1406_p * v1436_bc)) + (v1576_p * v1606_bc)) + (v1746_p * v1776_bc)) + (v1916_p * v1946_bc)) + (v2086_p * v2116_bc)) + (v2256_p * v2286_bc)) + (v2426_p * v2456_bc)) + (v2596_p * v2626_bc)) + (v2766_p * v2796_bc)) + (v2936_p * v2966_bc)) + (v3106_p * v3136_bc)) + (v3276_p * v3306_bc)) + (v3446_p * v3476_bc)) + (v3616_p * v3646_bc);
              double v3653_bc = sycl::select_from_group(item.get_sub_group(), v2802_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v3656_p = (((((((((((((((((((((v45_p * v82_bc) + (v216_p * v253_bc)) + (v386_p * v423_bc)) + (v556_p * v593_bc)) + (v726_p * v763_bc)) + (v896_p * v933_bc)) + (v1066_p * v1103_bc)) + (v1236_p * v1273_bc)) + (v1406_p * v1443_bc)) + (v1576_p * v1613_bc)) + (v1746_p * v1783_bc)) + (v1916_p * v1953_bc)) + (v2086_p * v2123_bc)) + (v2256_p * v2293_bc)) + (v2426_p * v2463_bc)) + (v2596_p * v2633_bc)) + (v2766_p * v2803_bc)) + (v2936_p * v2973_bc)) + (v3106_p * v3143_bc)) + (v3276_p * v3313_bc)) + (v3446_p * v3483_bc)) + (v3616_p * v3653_bc);
              double v3660_bc = sycl::select_from_group(item.get_sub_group(), v2809_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v3663_p = (((((((((((((((((((((v45_p * v89_bc) + (v216_p * v260_bc)) + (v386_p * v430_bc)) + (v556_p * v600_bc)) + (v726_p * v770_bc)) + (v896_p * v940_bc)) + (v1066_p * v1110_bc)) + (v1236_p * v1280_bc)) + (v1406_p * v1450_bc)) + (v1576_p * v1620_bc)) + (v1746_p * v1790_bc)) + (v1916_p * v1960_bc)) + (v2086_p * v2130_bc)) + (v2256_p * v2300_bc)) + (v2426_p * v2470_bc)) + (v2596_p * v2640_bc)) + (v2766_p * v2810_bc)) + (v2936_p * v2980_bc)) + (v3106_p * v3150_bc)) + (v3276_p * v3320_bc)) + (v3446_p * v3490_bc)) + (v3616_p * v3660_bc);
              double v3667_bc = sycl::select_from_group(item.get_sub_group(), v2816_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v3670_p = (((((((((((((((((((((v45_p * v96_bc) + (v216_p * v267_bc)) + (v386_p * v437_bc)) + (v556_p * v607_bc)) + (v726_p * v777_bc)) + (v896_p * v947_bc)) + (v1066_p * v1117_bc)) + (v1236_p * v1287_bc)) + (v1406_p * v1457_bc)) + (v1576_p * v1627_bc)) + (v1746_p * v1797_bc)) + (v1916_p * v1967_bc)) + (v2086_p * v2137_bc)) + (v2256_p * v2307_bc)) + (v2426_p * v2477_bc)) + (v2596_p * v2647_bc)) + (v2766_p * v2817_bc)) + (v2936_p * v2987_bc)) + (v3106_p * v3157_bc)) + (v3276_p * v3327_bc)) + (v3446_p * v3497_bc)) + (v3616_p * v3667_bc);
              double v3672_p = v3615_data[1];
              double v3677_p = (((((((((((((((((((((v101_p * v47_bc) + (v272_p * v218_bc)) + (v442_p * v388_bc)) + (v612_p * v558_bc)) + (v782_p * v728_bc)) + (v952_p * v898_bc)) + (v1122_p * v1068_bc)) + (v1292_p * v1238_bc)) + (v1462_p * v1408_bc)) + (v1632_p * v1578_bc)) + (v1802_p * v1748_bc)) + (v1972_p * v1918_bc)) + (v2142_p * v2088_bc)) + (v2312_p * v2258_bc)) + (v2482_p * v2428_bc)) + (v2652_p * v2598_bc)) + (v2822_p * v2768_bc)) + (v2992_p * v2938_bc)) + (v3162_p * v3108_bc)) + (v3332_p * v3278_bc)) + (v3502_p * v3448_bc)) + (v3672_p * v3618_bc);
              double v3684_p = (((((((((((((((((((((v101_p * v54_bc) + (v272_p * v225_bc)) + (v442_p * v395_bc)) + (v612_p * v565_bc)) + (v782_p * v735_bc)) + (v952_p * v905_bc)) + (v1122_p * v1075_bc)) + (v1292_p * v1245_bc)) + (v1462_p * v1415_bc)) + (v1632_p * v1585_bc)) + (v1802_p * v1755_bc)) + (v1972_p * v1925_bc)) + (v2142_p * v2095_bc)) + (v2312_p * v2265_bc)) + (v2482_p * v2435_bc)) + (v2652_p * v2605_bc)) + (v2822_p * v2775_bc)) + (v2992_p * v2945_bc)) + (v3162_p * v3115_bc)) + (v3332_p * v3285_bc)) + (v3502_p * v3455_bc)) + (v3672_p * v3625_bc);
              double v3691_p = (((((((((((((((((((((v101_p * v61_bc) + (v272_p * v232_bc)) + (v442_p * v402_bc)) + (v612_p * v572_bc)) + (v782_p * v742_bc)) + (v952_p * v912_bc)) + (v1122_p * v1082_bc)) + (v1292_p * v1252_bc)) + (v1462_p * v1422_bc)) + (v1632_p * v1592_bc)) + (v1802_p * v1762_bc)) + (v1972_p * v1932_bc)) + (v2142_p * v2102_bc)) + (v2312_p * v2272_bc)) + (v2482_p * v2442_bc)) + (v2652_p * v2612_bc)) + (v2822_p * v2782_bc)) + (v2992_p * v2952_bc)) + (v3162_p * v3122_bc)) + (v3332_p * v3292_bc)) + (v3502_p * v3462_bc)) + (v3672_p * v3632_bc);
              double v3698_p = (((((((((((((((((((((v101_p * v68_bc) + (v272_p * v239_bc)) + (v442_p * v409_bc)) + (v612_p * v579_bc)) + (v782_p * v749_bc)) + (v952_p * v919_bc)) + (v1122_p * v1089_bc)) + (v1292_p * v1259_bc)) + (v1462_p * v1429_bc)) + (v1632_p * v1599_bc)) + (v1802_p * v1769_bc)) + (v1972_p * v1939_bc)) + (v2142_p * v2109_bc)) + (v2312_p * v2279_bc)) + (v2482_p * v2449_bc)) + (v2652_p * v2619_bc)) + (v2822_p * v2789_bc)) + (v2992_p * v2959_bc)) + (v3162_p * v3129_bc)) + (v3332_p * v3299_bc)) + (v3502_p * v3469_bc)) + (v3672_p * v3639_bc);
              double v3705_p = (((((((((((((((((((((v101_p * v75_bc) + (v272_p * v246_bc)) + (v442_p * v416_bc)) + (v612_p * v586_bc)) + (v782_p * v756_bc)) + (v952_p * v926_bc)) + (v1122_p * v1096_bc)) + (v1292_p * v1266_bc)) + (v1462_p * v1436_bc)) + (v1632_p * v1606_bc)) + (v1802_p * v1776_bc)) + (v1972_p * v1946_bc)) + (v2142_p * v2116_bc)) + (v2312_p * v2286_bc)) + (v2482_p * v2456_bc)) + (v2652_p * v2626_bc)) + (v2822_p * v2796_bc)) + (v2992_p * v2966_bc)) + (v3162_p * v3136_bc)) + (v3332_p * v3306_bc)) + (v3502_p * v3476_bc)) + (v3672_p * v3646_bc);
              double v3712_p = (((((((((((((((((((((v101_p * v82_bc) + (v272_p * v253_bc)) + (v442_p * v423_bc)) + (v612_p * v593_bc)) + (v782_p * v763_bc)) + (v952_p * v933_bc)) + (v1122_p * v1103_bc)) + (v1292_p * v1273_bc)) + (v1462_p * v1443_bc)) + (v1632_p * v1613_bc)) + (v1802_p * v1783_bc)) + (v1972_p * v1953_bc)) + (v2142_p * v2123_bc)) + (v2312_p * v2293_bc)) + (v2482_p * v2463_bc)) + (v2652_p * v2633_bc)) + (v2822_p * v2803_bc)) + (v2992_p * v2973_bc)) + (v3162_p * v3143_bc)) + (v3332_p * v3313_bc)) + (v3502_p * v3483_bc)) + (v3672_p * v3653_bc);
              double v3719_p = (((((((((((((((((((((v101_p * v89_bc) + (v272_p * v260_bc)) + (v442_p * v430_bc)) + (v612_p * v600_bc)) + (v782_p * v770_bc)) + (v952_p * v940_bc)) + (v1122_p * v1110_bc)) + (v1292_p * v1280_bc)) + (v1462_p * v1450_bc)) + (v1632_p * v1620_bc)) + (v1802_p * v1790_bc)) + (v1972_p * v1960_bc)) + (v2142_p * v2130_bc)) + (v2312_p * v2300_bc)) + (v2482_p * v2470_bc)) + (v2652_p * v2640_bc)) + (v2822_p * v2810_bc)) + (v2992_p * v2980_bc)) + (v3162_p * v3150_bc)) + (v3332_p * v3320_bc)) + (v3502_p * v3490_bc)) + (v3672_p * v3660_bc);
              double v3726_p = (((((((((((((((((((((v101_p * v96_bc) + (v272_p * v267_bc)) + (v442_p * v437_bc)) + (v612_p * v607_bc)) + (v782_p * v777_bc)) + (v952_p * v947_bc)) + (v1122_p * v1117_bc)) + (v1292_p * v1287_bc)) + (v1462_p * v1457_bc)) + (v1632_p * v1627_bc)) + (v1802_p * v1797_bc)) + (v1972_p * v1967_bc)) + (v2142_p * v2137_bc)) + (v2312_p * v2307_bc)) + (v2482_p * v2477_bc)) + (v2652_p * v2647_bc)) + (v2822_p * v2817_bc)) + (v2992_p * v2987_bc)) + (v3162_p * v3157_bc)) + (v3332_p * v3327_bc)) + (v3502_p * v3497_bc)) + (v3672_p * v3667_bc);
              sycl::vec<double, 2> v3728_data = *(sycl::vec<double, 2>*)&glb_m1[(1344 + v156_a)];
              double v3729_p = v3728_data[0];
              double v3734_p = (((((((((((((((((((((v159_p * v47_bc) + (v329_p * v218_bc)) + (v499_p * v388_bc)) + (v669_p * v558_bc)) + (v839_p * v728_bc)) + (v1009_p * v898_bc)) + (v1179_p * v1068_bc)) + (v1349_p * v1238_bc)) + (v1519_p * v1408_bc)) + (v1689_p * v1578_bc)) + (v1859_p * v1748_bc)) + (v2029_p * v1918_bc)) + (v2199_p * v2088_bc)) + (v2369_p * v2258_bc)) + (v2539_p * v2428_bc)) + (v2709_p * v2598_bc)) + (v2879_p * v2768_bc)) + (v3049_p * v2938_bc)) + (v3219_p * v3108_bc)) + (v3389_p * v3278_bc)) + (v3559_p * v3448_bc)) + (v3729_p * v3618_bc);
              double v3741_p = (((((((((((((((((((((v159_p * v54_bc) + (v329_p * v225_bc)) + (v499_p * v395_bc)) + (v669_p * v565_bc)) + (v839_p * v735_bc)) + (v1009_p * v905_bc)) + (v1179_p * v1075_bc)) + (v1349_p * v1245_bc)) + (v1519_p * v1415_bc)) + (v1689_p * v1585_bc)) + (v1859_p * v1755_bc)) + (v2029_p * v1925_bc)) + (v2199_p * v2095_bc)) + (v2369_p * v2265_bc)) + (v2539_p * v2435_bc)) + (v2709_p * v2605_bc)) + (v2879_p * v2775_bc)) + (v3049_p * v2945_bc)) + (v3219_p * v3115_bc)) + (v3389_p * v3285_bc)) + (v3559_p * v3455_bc)) + (v3729_p * v3625_bc);
              double v3746_p = v3729_p * v3632_bc;
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v4187_i0 = 0; v4187_i0 < 1; ++v4187_i0) {
                int32_t v4189_lead = v4147_lead + (v4187_i0 * 16);
                #pragma unroll
                for (int32_t v4190_i1 = 0; v4190_i1 < 8; ++v4190_i1) {
                  double v4193_data = wrap_glb_m2[(v4189_lead + (v4190_i1 * 24))];
                  r0[(v4187_i0 + (v4190_i1 * 2))] = v4193_data;
                }
              }
              #pragma unroll
              for (int32_t v4196_i1 = 0; v4196_i1 < 8; ++v4196_i1) {
                double v4199_data_pre = wrap_glb_m2[v4150_g ? ((v4149_lead + (v4196_i1 * 24))) : (0)];
                double v4199_data = v4150_g ? (v4199_data_pre) : (0.0);
                r0[(1 + (v4196_i1 * 2))] = v4199_data;
              }
              sycl::vec<double, 2> v3785_data = *(sycl::vec<double, 2>*)&glb_m1[(1408 + v42_a)];
              double v3786_p = v3785_data[0];
              double v3788_bc = sycl::select_from_group(item.get_sub_group(), v2767_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v3795_bc = sycl::select_from_group(item.get_sub_group(), v2774_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v3798_p = v3628_p + (v3786_p * v3795_bc);
              double v3802_bc = sycl::select_from_group(item.get_sub_group(), v2781_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v3805_p = v3635_p + (v3786_p * v3802_bc);
              double v3809_bc = sycl::select_from_group(item.get_sub_group(), v2788_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v3812_p = v3642_p + (v3786_p * v3809_bc);
              double v3816_bc = sycl::select_from_group(item.get_sub_group(), v2795_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v3819_p = v3649_p + (v3786_p * v3816_bc);
              double v3823_bc = sycl::select_from_group(item.get_sub_group(), v2802_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v3826_p = v3656_p + (v3786_p * v3823_bc);
              double v3830_bc = sycl::select_from_group(item.get_sub_group(), v2809_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v3833_p = v3663_p + (v3786_p * v3830_bc);
              double v3837_bc = sycl::select_from_group(item.get_sub_group(), v2816_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v3840_p = v3670_p + (v3786_p * v3837_bc);
              double v3842_p = v3785_data[1];
              double v3847_p = v3677_p + (v3842_p * v3788_bc);
              double v3854_p = v3684_p + (v3842_p * v3795_bc);
              double v3861_p = v3691_p + (v3842_p * v3802_bc);
              double v3868_p = v3698_p + (v3842_p * v3809_bc);
              double v3875_p = v3705_p + (v3842_p * v3816_bc);
              double v3882_p = v3712_p + (v3842_p * v3823_bc);
              double v3889_p = v3719_p + (v3842_p * v3830_bc);
              double v3896_p = v3726_p + (v3842_p * v3837_bc);
              sycl::vec<double, 2> v3898_data = *(sycl::vec<double, 2>*)&glb_m1[(1408 + v156_a)];
              double v3899_p = v3898_data[0];
              double v3904_p = v3734_p + (v3899_p * v3788_bc);
              double v3911_p = v3741_p + (v3899_p * v3795_bc);
              double v3918_p = (v3578_p + v3746_p) + (v3899_p * v3802_bc);
              double v3925_p = (v3585_p + (v3729_p * v3639_bc)) + (v3899_p * v3809_bc);
              double v3932_p = (v3592_p + (v3729_p * v3646_bc)) + (v3899_p * v3816_bc);
              double v3939_p = (v3599_p + (v3729_p * v3653_bc)) + (v3899_p * v3823_bc);
              double v3946_p = (v3606_p + (v3729_p * v3660_bc)) + (v3899_p * v3830_bc);
              double v3953_p = (v3613_p + (v3729_p * v3667_bc)) + (v3899_p * v3837_bc);
              sycl::vec<double, 2> v3955_data = *(sycl::vec<double, 2>*)&glb_m1[(1472 + v42_a)];
              double v3956_p = v3955_data[0];
              double v3958_bc = sycl::select_from_group(item.get_sub_group(), v2767_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              ir1[0] = ((v3621_p + (v3786_p * v3788_bc)) + (v3956_p * v3958_bc));
              double v3965_bc = sycl::select_from_group(item.get_sub_group(), v2774_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              ir1[3] = (v3798_p + (v3956_p * v3965_bc));
              double v3972_bc = sycl::select_from_group(item.get_sub_group(), v2781_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              ir1[6] = (v3805_p + (v3956_p * v3972_bc));
              double v3979_bc = sycl::select_from_group(item.get_sub_group(), v2788_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              ir1[9] = (v3812_p + (v3956_p * v3979_bc));
              double v3986_bc = sycl::select_from_group(item.get_sub_group(), v2795_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              ir1[12] = (v3819_p + (v3956_p * v3986_bc));
              double v3993_bc = sycl::select_from_group(item.get_sub_group(), v2802_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              ir1[15] = (v3826_p + (v3956_p * v3993_bc));
              double v4000_bc = sycl::select_from_group(item.get_sub_group(), v2809_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              ir1[18] = (v3833_p + (v3956_p * v4000_bc));
              double v4007_bc = sycl::select_from_group(item.get_sub_group(), v2816_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              ir1[21] = (v3840_p + (v3956_p * v4007_bc));
              double v4012_p = v3955_data[1];
              ir1[1] = (v3847_p + (v4012_p * v3958_bc));
              ir1[4] = (v3854_p + (v4012_p * v3965_bc));
              ir1[7] = (v3861_p + (v4012_p * v3972_bc));
              ir1[10] = (v3868_p + (v4012_p * v3979_bc));
              ir1[13] = (v3875_p + (v4012_p * v3986_bc));
              ir1[16] = (v3882_p + (v4012_p * v3993_bc));
              ir1[19] = (v3889_p + (v4012_p * v4000_bc));
              ir1[22] = (v3896_p + (v4012_p * v4007_bc));
              sycl::vec<double, 2> v4068_data = *(sycl::vec<double, 2>*)&glb_m1[(1472 + v156_a)];
              double v4069_p = v4068_data[0];
              ir1[2] = (v3904_p + (v4069_p * v3958_bc));
              ir1[5] = (v3911_p + (v4069_p * v3965_bc));
              ir1[8] = (v3918_p + (v4069_p * v3972_bc));
              ir1[11] = (v3925_p + (v4069_p * v3979_bc));
              ir1[14] = (v3932_p + (v4069_p * v3986_bc));
              ir1[17] = (v3939_p + (v4069_p * v3993_bc));
              ir1[20] = (v3946_p + (v4069_p * v4000_bc));
              ir1[23] = (v3953_p + (v4069_p * v4007_bc));
              // r1 = ir1
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v4129_i0 = 0; v4129_i0 < 3; ++v4129_i0) {
                int32_t v4135_lead = v4147_lead + (v4129_i0 * 16);
                #pragma unroll
                for (int32_t v4130_i1 = 0; v4130_i1 < 8; ++v4130_i1) {
                  double v4133_data = ir1[(v4129_i0 + (v4130_i1 * 3))];
                  glb_m0[(v4135_lead + (v4130_i1 * 48))] = v4133_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
            else {
              // r0 = load{g>r}(wrap_glb_m2);
              #pragma unroll
              for (int32_t v4157_i0 = 0; v4157_i0 < 1; ++v4157_i0) {
                int32_t v4159_lead = v4147_lead + (v4157_i0 * 16);
                #pragma unroll
                for (int32_t v4160_i1 = 0; v4160_i1 < 8; ++v4160_i1) {
                  double v4163_data = wrap_glb_m2[(v4159_lead + (v4160_i1 * 24))];
                  r0[(v4157_i0 + (v4160_i1 * 2))] = v4163_data;
                }
              }
              #pragma unroll
              for (int32_t v4166_i1 = 0; v4166_i1 < 8; ++v4166_i1) {
                double v4169_data_pre = wrap_glb_m2[v4150_g ? ((v4149_lead + (v4166_i1 * 24))) : (0)];
                double v4169_data = v4150_g ? (v4169_data_pre) : (0.0);
                r0[(1 + (v4166_i1 * 2))] = v4169_data;
              }
            }
            v4143_flagWord = v4144_flagWordNext;
            v4144_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}

