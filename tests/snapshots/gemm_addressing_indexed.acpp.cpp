// === base name ===
kernel_3068eb54db2aa80f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3068eb54db2aa80f = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3068eb54db2aa80f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3068eb54db2aa80f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3068eb54db2aa80f(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_3068eb54db2aa80f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3068eb54db2aa80f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3068eb54db2aa80f(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m1_stride, m2, m2_extraOffset, m2_slot, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3068eb54db2aa80f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided(stride=300, offset=8)
        //   m1 16×16(16×16) {0..16}×{0..16} indexed(stride=runtime)
        //   m2 16×16(16×16) {0..16}×{0..16} indexed_pointer(offset=4, slots=4, slot=runtime)
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided(stride=300, offset=8)","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed(stride=runtime)","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed_pointer(offset=4, slots=4, slot=runtime)","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided(stride=300, offset=8)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"indexed(stride=runtime)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"indexed_pointer(offset=4, slots=4, slot=runtime)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          float r0[16]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v1590_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v1590_flagSecondAhead < numElements0) ? v1590_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const float *const __restrict__ peel_glb_m1 = &m1[static_cast<size_t>(elementIndices[v3_batchId1]) * m1_stride + 0 + m1_extraOffset];
          int32_t v1600_lead = item.get_local_id(2) % 16;
          // r0 = load{g>r}(peel_glb_m1);
          #pragma unroll
          for (int32_t v1612_i0 = 0; v1612_i0 < 1; ++v1612_i0) {
            int32_t v1614_lead = v1600_lead + (v1612_i0 * 16);
            #pragma unroll
            for (int32_t v1615_i1 = 0; v1615_i1 < 16; ++v1615_i1) {
              float v1618_data_pre = peel_glb_m1[allowed_peel ? ((v1614_lead + (v1615_i1 * 16))) : (0)];
              float v1618_data = allowed_peel ? (v1618_data_pre) : (0.0f);
              r0[(v1612_i0 + v1615_i1)] = v1618_data;
            }
          }
          uint32_t v1594_flagWord = flagWordFirst;
          uint32_t v1595_flagWordNext = flagWordSecond;
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const float *const __restrict__ wrap_glb_m1 = &m1[static_cast<size_t>(elementIndices[v10_batchId1]) * m1_stride + 0 + m1_extraOffset];
            size_t v1630_flagAheadAhead = v10_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v1630_flagAheadAhead < numElements0) ? v1630_flagAheadAhead : v10_batchId1)];
            const bool allowed = static_cast<bool>(v1594_flagWord);
            const bool allowed_next = static_cast<bool>(v1595_flagWordNext);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 300 + 0 + 8 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[static_cast<size_t>(elementIndices[v7_batchId0]) * 4 + m2_slot][0 + 4 + m2_extraOffset];
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
                int32_t v34_lead = v1600_lead + (v31_i0 * 16);
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 16; ++v32_i1) {
                  float v37_data = glb_m2[(v34_lead + (v32_i1 * 16))];
                  r1[(v31_i0 + v32_i1)] = v37_data;
                }
              }
              float r2[16]{};
              // ir2 = +(r0 * r1)
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir2[16]{};
              float v41_data = r0[0];
              float v42_data = r1[0];
              float v43_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v48_data = r1[1];
              float v49_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v54_data = r1[2];
              float v55_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v60_data = r1[3];
              float v61_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v66_data = r1[4];
              float v67_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v72_data = r1[5];
              float v73_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v78_data = r1[6];
              float v79_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v84_data = r1[7];
              float v85_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v90_data = r1[8];
              float v91_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v96_data = r1[9];
              float v97_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v102_data = r1[10];
              float v103_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v108_data = r1[11];
              float v109_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v114_data = r1[12];
              float v115_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v120_data = r1[13];
              float v121_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v126_data = r1[14];
              float v127_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v132_data = r1[15];
              float v133_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v137_data = r0[1];
              float v139_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v145_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v151_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v157_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v163_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v169_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v175_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v181_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v187_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v193_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v199_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v205_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v211_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v217_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v223_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v229_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v233_data = r0[2];
              float v235_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v241_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v247_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v253_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v259_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v265_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v271_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v277_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v283_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v289_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v295_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v301_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v307_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v313_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v319_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v325_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v329_data = r0[3];
              float v331_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v337_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v343_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v349_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v355_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v361_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v367_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v373_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v379_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v385_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v391_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v397_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v403_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v409_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v415_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v421_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v425_data = r0[4];
              float v427_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v433_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v439_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v445_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v451_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v457_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v463_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v469_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v475_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v481_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v487_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v493_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v499_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v505_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v511_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v517_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v521_data = r0[5];
              float v523_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v529_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v535_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v541_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v547_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v553_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v559_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v565_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v571_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v577_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v583_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v589_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v595_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v601_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v607_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v613_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v617_data = r0[6];
              float v619_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v625_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v631_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v637_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v643_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v649_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v655_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v661_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v667_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v673_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v679_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v685_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v691_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v697_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v703_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v709_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v713_data = r0[7];
              float v715_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v721_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v727_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v733_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v739_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v745_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v751_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v757_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v763_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v769_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v775_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v781_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v787_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v793_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v799_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v805_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v809_data = r0[8];
              float v811_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v817_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v823_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v829_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v835_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v841_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v847_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v853_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v859_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v865_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v871_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v877_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v883_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v889_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v895_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v901_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v905_data = r0[9];
              float v907_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v913_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v919_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v925_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v931_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v937_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v943_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v949_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v955_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v961_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v967_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v973_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v979_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v985_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v991_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v997_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v1001_data = r0[10];
              float v1003_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1009_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1015_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1021_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1027_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1033_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1039_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1045_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1051_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1057_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1063_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1069_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1075_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1081_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1087_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1093_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v1097_data = r0[11];
              float v1099_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1105_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1111_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1117_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1123_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1129_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1135_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1141_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1147_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1153_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1159_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1165_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1171_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1177_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1183_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1189_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v1193_data = r0[12];
              float v1195_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1201_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1207_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1213_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1219_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1225_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1231_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1237_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1243_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1249_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1255_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1261_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1267_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1273_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1279_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1285_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v1289_data = r0[13];
              float v1291_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1297_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1303_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1309_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1315_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1321_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1327_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1333_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1339_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1345_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1351_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1357_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1363_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1369_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1375_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1381_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v1385_data = r0[14];
              float v1387_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1390_p = ((((((((((((((v41_data * v43_bc) + (v137_data * v139_bc)) + (v233_data * v235_bc)) + (v329_data * v331_bc)) + (v425_data * v427_bc)) + (v521_data * v523_bc)) + (v617_data * v619_bc)) + (v713_data * v715_bc)) + (v809_data * v811_bc)) + (v905_data * v907_bc)) + (v1001_data * v1003_bc)) + (v1097_data * v1099_bc)) + (v1193_data * v1195_bc)) + (v1289_data * v1291_bc)) + (v1385_data * v1387_bc);
              float v1393_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1396_p = ((((((((((((((v41_data * v49_bc) + (v137_data * v145_bc)) + (v233_data * v241_bc)) + (v329_data * v337_bc)) + (v425_data * v433_bc)) + (v521_data * v529_bc)) + (v617_data * v625_bc)) + (v713_data * v721_bc)) + (v809_data * v817_bc)) + (v905_data * v913_bc)) + (v1001_data * v1009_bc)) + (v1097_data * v1105_bc)) + (v1193_data * v1201_bc)) + (v1289_data * v1297_bc)) + (v1385_data * v1393_bc);
              float v1399_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1402_p = ((((((((((((((v41_data * v55_bc) + (v137_data * v151_bc)) + (v233_data * v247_bc)) + (v329_data * v343_bc)) + (v425_data * v439_bc)) + (v521_data * v535_bc)) + (v617_data * v631_bc)) + (v713_data * v727_bc)) + (v809_data * v823_bc)) + (v905_data * v919_bc)) + (v1001_data * v1015_bc)) + (v1097_data * v1111_bc)) + (v1193_data * v1207_bc)) + (v1289_data * v1303_bc)) + (v1385_data * v1399_bc);
              float v1405_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1408_p = ((((((((((((((v41_data * v61_bc) + (v137_data * v157_bc)) + (v233_data * v253_bc)) + (v329_data * v349_bc)) + (v425_data * v445_bc)) + (v521_data * v541_bc)) + (v617_data * v637_bc)) + (v713_data * v733_bc)) + (v809_data * v829_bc)) + (v905_data * v925_bc)) + (v1001_data * v1021_bc)) + (v1097_data * v1117_bc)) + (v1193_data * v1213_bc)) + (v1289_data * v1309_bc)) + (v1385_data * v1405_bc);
              float v1411_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1414_p = ((((((((((((((v41_data * v67_bc) + (v137_data * v163_bc)) + (v233_data * v259_bc)) + (v329_data * v355_bc)) + (v425_data * v451_bc)) + (v521_data * v547_bc)) + (v617_data * v643_bc)) + (v713_data * v739_bc)) + (v809_data * v835_bc)) + (v905_data * v931_bc)) + (v1001_data * v1027_bc)) + (v1097_data * v1123_bc)) + (v1193_data * v1219_bc)) + (v1289_data * v1315_bc)) + (v1385_data * v1411_bc);
              float v1417_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1420_p = ((((((((((((((v41_data * v73_bc) + (v137_data * v169_bc)) + (v233_data * v265_bc)) + (v329_data * v361_bc)) + (v425_data * v457_bc)) + (v521_data * v553_bc)) + (v617_data * v649_bc)) + (v713_data * v745_bc)) + (v809_data * v841_bc)) + (v905_data * v937_bc)) + (v1001_data * v1033_bc)) + (v1097_data * v1129_bc)) + (v1193_data * v1225_bc)) + (v1289_data * v1321_bc)) + (v1385_data * v1417_bc);
              float v1423_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1426_p = ((((((((((((((v41_data * v79_bc) + (v137_data * v175_bc)) + (v233_data * v271_bc)) + (v329_data * v367_bc)) + (v425_data * v463_bc)) + (v521_data * v559_bc)) + (v617_data * v655_bc)) + (v713_data * v751_bc)) + (v809_data * v847_bc)) + (v905_data * v943_bc)) + (v1001_data * v1039_bc)) + (v1097_data * v1135_bc)) + (v1193_data * v1231_bc)) + (v1289_data * v1327_bc)) + (v1385_data * v1423_bc);
              float v1429_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1432_p = ((((((((((((((v41_data * v85_bc) + (v137_data * v181_bc)) + (v233_data * v277_bc)) + (v329_data * v373_bc)) + (v425_data * v469_bc)) + (v521_data * v565_bc)) + (v617_data * v661_bc)) + (v713_data * v757_bc)) + (v809_data * v853_bc)) + (v905_data * v949_bc)) + (v1001_data * v1045_bc)) + (v1097_data * v1141_bc)) + (v1193_data * v1237_bc)) + (v1289_data * v1333_bc)) + (v1385_data * v1429_bc);
              float v1435_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1438_p = ((((((((((((((v41_data * v91_bc) + (v137_data * v187_bc)) + (v233_data * v283_bc)) + (v329_data * v379_bc)) + (v425_data * v475_bc)) + (v521_data * v571_bc)) + (v617_data * v667_bc)) + (v713_data * v763_bc)) + (v809_data * v859_bc)) + (v905_data * v955_bc)) + (v1001_data * v1051_bc)) + (v1097_data * v1147_bc)) + (v1193_data * v1243_bc)) + (v1289_data * v1339_bc)) + (v1385_data * v1435_bc);
              float v1441_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1444_p = ((((((((((((((v41_data * v97_bc) + (v137_data * v193_bc)) + (v233_data * v289_bc)) + (v329_data * v385_bc)) + (v425_data * v481_bc)) + (v521_data * v577_bc)) + (v617_data * v673_bc)) + (v713_data * v769_bc)) + (v809_data * v865_bc)) + (v905_data * v961_bc)) + (v1001_data * v1057_bc)) + (v1097_data * v1153_bc)) + (v1193_data * v1249_bc)) + (v1289_data * v1345_bc)) + (v1385_data * v1441_bc);
              float v1447_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1450_p = ((((((((((((((v41_data * v103_bc) + (v137_data * v199_bc)) + (v233_data * v295_bc)) + (v329_data * v391_bc)) + (v425_data * v487_bc)) + (v521_data * v583_bc)) + (v617_data * v679_bc)) + (v713_data * v775_bc)) + (v809_data * v871_bc)) + (v905_data * v967_bc)) + (v1001_data * v1063_bc)) + (v1097_data * v1159_bc)) + (v1193_data * v1255_bc)) + (v1289_data * v1351_bc)) + (v1385_data * v1447_bc);
              float v1453_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1456_p = ((((((((((((((v41_data * v109_bc) + (v137_data * v205_bc)) + (v233_data * v301_bc)) + (v329_data * v397_bc)) + (v425_data * v493_bc)) + (v521_data * v589_bc)) + (v617_data * v685_bc)) + (v713_data * v781_bc)) + (v809_data * v877_bc)) + (v905_data * v973_bc)) + (v1001_data * v1069_bc)) + (v1097_data * v1165_bc)) + (v1193_data * v1261_bc)) + (v1289_data * v1357_bc)) + (v1385_data * v1453_bc);
              float v1459_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1462_p = ((((((((((((((v41_data * v115_bc) + (v137_data * v211_bc)) + (v233_data * v307_bc)) + (v329_data * v403_bc)) + (v425_data * v499_bc)) + (v521_data * v595_bc)) + (v617_data * v691_bc)) + (v713_data * v787_bc)) + (v809_data * v883_bc)) + (v905_data * v979_bc)) + (v1001_data * v1075_bc)) + (v1097_data * v1171_bc)) + (v1193_data * v1267_bc)) + (v1289_data * v1363_bc)) + (v1385_data * v1459_bc);
              float v1465_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1468_p = ((((((((((((((v41_data * v121_bc) + (v137_data * v217_bc)) + (v233_data * v313_bc)) + (v329_data * v409_bc)) + (v425_data * v505_bc)) + (v521_data * v601_bc)) + (v617_data * v697_bc)) + (v713_data * v793_bc)) + (v809_data * v889_bc)) + (v905_data * v985_bc)) + (v1001_data * v1081_bc)) + (v1097_data * v1177_bc)) + (v1193_data * v1273_bc)) + (v1289_data * v1369_bc)) + (v1385_data * v1465_bc);
              float v1471_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1474_p = ((((((((((((((v41_data * v127_bc) + (v137_data * v223_bc)) + (v233_data * v319_bc)) + (v329_data * v415_bc)) + (v425_data * v511_bc)) + (v521_data * v607_bc)) + (v617_data * v703_bc)) + (v713_data * v799_bc)) + (v809_data * v895_bc)) + (v905_data * v991_bc)) + (v1001_data * v1087_bc)) + (v1097_data * v1183_bc)) + (v1193_data * v1279_bc)) + (v1289_data * v1375_bc)) + (v1385_data * v1471_bc);
              float v1477_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v1480_p = ((((((((((((((v41_data * v133_bc) + (v137_data * v229_bc)) + (v233_data * v325_bc)) + (v329_data * v421_bc)) + (v425_data * v517_bc)) + (v521_data * v613_bc)) + (v617_data * v709_bc)) + (v713_data * v805_bc)) + (v809_data * v901_bc)) + (v905_data * v997_bc)) + (v1001_data * v1093_bc)) + (v1097_data * v1189_bc)) + (v1193_data * v1285_bc)) + (v1289_data * v1381_bc)) + (v1385_data * v1477_bc);
              float v1481_data = r0[15];
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v1620_i0 = 0; v1620_i0 < 1; ++v1620_i0) {
                int32_t v1622_lead = v1600_lead + (v1620_i0 * 16);
                #pragma unroll
                for (int32_t v1623_i1 = 0; v1623_i1 < 16; ++v1623_i1) {
                  float v1626_data_pre = wrap_glb_m1[allowed_next ? ((v1622_lead + (v1623_i1 * 16))) : (0)];
                  float v1626_data = allowed_next ? (v1626_data_pre) : (0.0f);
                  r0[(v1620_i0 + v1623_i1)] = v1626_data;
                }
              }
              float v1483_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[0] = (v1390_p + (v1481_data * v1483_bc));
              float v1489_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[1] = (v1396_p + (v1481_data * v1489_bc));
              float v1495_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[2] = (v1402_p + (v1481_data * v1495_bc));
              float v1501_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[3] = (v1408_p + (v1481_data * v1501_bc));
              float v1507_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[4] = (v1414_p + (v1481_data * v1507_bc));
              float v1513_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[5] = (v1420_p + (v1481_data * v1513_bc));
              float v1519_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[6] = (v1426_p + (v1481_data * v1519_bc));
              float v1525_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[7] = (v1432_p + (v1481_data * v1525_bc));
              float v1531_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[8] = (v1438_p + (v1481_data * v1531_bc));
              float v1537_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[9] = (v1444_p + (v1481_data * v1537_bc));
              float v1543_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[10] = (v1450_p + (v1481_data * v1543_bc));
              float v1549_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[11] = (v1456_p + (v1481_data * v1549_bc));
              float v1555_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[12] = (v1462_p + (v1481_data * v1555_bc));
              float v1561_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[13] = (v1468_p + (v1481_data * v1561_bc));
              float v1567_bc = sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[14] = (v1474_p + (v1481_data * v1567_bc));
              float v1573_bc = sycl::select_from_group(item.get_sub_group(), v132_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              ir2[15] = (v1480_p + (v1481_data * v1573_bc));
              // r2 = ir2
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1581_i0 = 0; v1581_i0 < 1; ++v1581_i0) {
                int32_t v1586_lead = v1600_lead + (v1581_i0 * 16);
                #pragma unroll
                for (int32_t v1582_i1 = 0; v1582_i1 < 16; ++v1582_i1) {
                  float v1584_data = ir2[v1582_i1];
                  glb_m0[(v1586_lead + (v1582_i1 * 16))] = v1584_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
            else {
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v1604_i0 = 0; v1604_i0 < 1; ++v1604_i0) {
                int32_t v1606_lead = v1600_lead + (v1604_i0 * 16);
                #pragma unroll
                for (int32_t v1607_i1 = 0; v1607_i1 < 16; ++v1607_i1) {
                  float v1610_data_pre = wrap_glb_m1[allowed_next ? ((v1606_lead + (v1607_i1 * 16))) : (0)];
                  float v1610_data = allowed_next ? (v1610_data_pre) : (0.0f);
                  r0[(v1604_i0 + v1607_i1)] = v1610_data;
                }
              }
            }
            v1594_flagWord = v1595_flagWordNext;
            v1595_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}

