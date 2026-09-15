// === base name ===
kernel_f351ec2def0c1604

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f351ec2def0c1604 = {{1, 8, 1}, 32, 40, 1, 8, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f351ec2def0c1604(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f351ec2def0c1604(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f351ec2def0c1604(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 384 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_f351ec2def0c1604(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f351ec2def0c1604(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f351ec2def0c1604(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f351ec2def0c1604(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<384 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (40 active) x 8 per block = block 1x8x1, 1536 B shared, occupancy grid
        // operands:
        //   m0 40×6(40×6) {0..40}×{0..6} strided
        //   m1 40×8(40×8) {0..40}×{0..8} none
        //   m2 8×6(8×6) {0..8}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":384}],"shared_bytes":1536,"shared_elements":384,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":false,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (48 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (48);
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const float *const __restrict__ pf_glb_m2 = &m2[v9_batchId1 * 48 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 240 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 48 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v17_ld;
              v17_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v17_ld);
              tensorforge::intel_esimd::simd<float, 16> v18_ld;
              v18_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 32));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v18_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 384> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 40), (0, 6)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 384> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v25_data(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v25_data_part;
              v25_data_part.copy_from(glb_m1 + (32_i32));
              v25_data.template select<8, 1>(0) = v25_data_part;
              tensorforge::intel_esimd::simd<float, 48> s0_w0 = tensorforge::slmLoad<float, 48>(s0 + 0);
              float v26_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v28_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v28_data + (v25_data * v26_data));
              float v31_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v33_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v33_data + (v25_data * v31_data));
              float v36_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v38_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v38_data + (v25_data * v36_data));
              float v41_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v43_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v43_data + (v25_data * v41_data));
              float v46_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 32> v48_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v48_data + (v25_data * v46_data));
              float v51_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 32> v53_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v53_data + (v25_data * v51_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0;
              glb_m1_run0.copy_from(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 32> v56_data(glb_m1_run0.template select<32, 1>(0));
              float v57_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v59_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v59_data + (v56_data * v57_data));
              float v62_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v64_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v64_data + (v56_data * v62_data));
              float v67_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v69_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v69_data + (v56_data * v67_data));
              float v72_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v74_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v74_data + (v56_data * v72_data));
              float v77_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 32> v79_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v79_data + (v56_data * v77_data));
              float v82_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v84_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v84_data + (v56_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v87_data(glb_m1_run0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v90_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v90_data + (v87_data * v57_data));
              tensorforge::intel_esimd::simd<float, 32> v95_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v95_data + (v87_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v100_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v100_data + (v87_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v105_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v105_data + (v87_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v110_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v110_data + (v87_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v115_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v115_data + (v87_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v118_data(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v118_data_part;
              v118_data_part.copy_from(glb_m1 + (112_i32));
              v118_data.template select<8, 1>(0) = v118_data_part;
              float v119_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v121_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v121_data + (v118_data * v119_data));
              float v124_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v126_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v126_data + (v118_data * v124_data));
              float v129_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v131_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v131_data + (v118_data * v129_data));
              float v134_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v136_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v136_data + (v118_data * v134_data));
              float v139_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v141_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v141_data + (v118_data * v139_data));
              float v144_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 32> v146_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v146_data + (v118_data * v144_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1;
              glb_m1_run1.copy_from(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 32> v149_data(glb_m1_run1.template select<32, 1>(0));
              float v150_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v152_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v152_data + (v149_data * v150_data));
              float v155_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v157_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v157_data + (v149_data * v155_data));
              float v160_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v162_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v162_data + (v149_data * v160_data));
              float v165_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v167_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v167_data + (v149_data * v165_data));
              float v170_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v172_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v172_data + (v149_data * v170_data));
              float v175_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v177_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v177_data + (v149_data * v175_data));
              tensorforge::intel_esimd::simd<float, 32> v180_data(glb_m1_run1.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v183_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v183_data + (v180_data * v150_data));
              tensorforge::intel_esimd::simd<float, 32> v188_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v188_data + (v180_data * v155_data));
              tensorforge::intel_esimd::simd<float, 32> v193_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v193_data + (v180_data * v160_data));
              tensorforge::intel_esimd::simd<float, 32> v198_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v198_data + (v180_data * v165_data));
              tensorforge::intel_esimd::simd<float, 32> v203_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v203_data + (v180_data * v170_data));
              tensorforge::intel_esimd::simd<float, 32> v208_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v208_data + (v180_data * v175_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2;
              glb_m1_run2.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 32> v211_data(glb_m1_run2.template select<32, 1>(0));
              float v212_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v214_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v214_data + (v211_data * v212_data));
              float v217_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v219_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v219_data + (v211_data * v217_data));
              float v222_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v224_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v224_data + (v211_data * v222_data));
              float v227_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v229_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v229_data + (v211_data * v227_data));
              float v232_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v234_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v234_data + (v211_data * v232_data));
              float v237_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v239_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v239_data + (v211_data * v237_data));
              tensorforge::intel_esimd::simd<float, 32> v242_data(glb_m1_run2.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v245_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v245_data + (v242_data * v212_data));
              tensorforge::intel_esimd::simd<float, 32> v250_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v250_data + (v242_data * v217_data));
              tensorforge::intel_esimd::simd<float, 32> v255_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v255_data + (v242_data * v222_data));
              tensorforge::intel_esimd::simd<float, 32> v260_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v260_data + (v242_data * v227_data));
              tensorforge::intel_esimd::simd<float, 32> v265_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v265_data + (v242_data * v232_data));
              tensorforge::intel_esimd::simd<float, 32> v270_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v270_data + (v242_data * v237_data));
              tensorforge::intel_esimd::simd<float, 32> v273_data(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v273_data_part;
              v273_data_part.copy_from(glb_m1 + (232_i32));
              v273_data.template select<8, 1>(0) = v273_data_part;
              float v274_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v276_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v276_data + (v273_data * v274_data));
              float v279_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v281_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v281_data + (v273_data * v279_data));
              float v284_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v286_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v286_data + (v273_data * v284_data));
              float v289_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v291_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v291_data + (v273_data * v289_data));
              float v294_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v296_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v296_data + (v273_data * v294_data));
              float v299_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v301_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v301_data + (v273_data * v299_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3;
              glb_m1_run3.copy_from(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 32> v304_data(glb_m1_run3.template select<32, 1>(0));
              float v305_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v307_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v307_data + (v304_data * v305_data));
              float v310_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v312_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v312_data + (v304_data * v310_data));
              float v315_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v317_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v317_data + (v304_data * v315_data));
              float v320_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v322_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v322_data + (v304_data * v320_data));
              float v325_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v327_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v327_data + (v304_data * v325_data));
              float v330_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v332_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v332_data + (v304_data * v330_data));
              tensorforge::intel_esimd::simd<float, 32> v335_data(glb_m1_run3.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v338_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v338_data + (v335_data * v305_data));
              tensorforge::intel_esimd::simd<float, 32> v343_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v343_data + (v335_data * v310_data));
              tensorforge::intel_esimd::simd<float, 32> v348_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v348_data + (v335_data * v315_data));
              tensorforge::intel_esimd::simd<float, 32> v353_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v353_data + (v335_data * v320_data));
              tensorforge::intel_esimd::simd<float, 32> v358_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v358_data + (v335_data * v325_data));
              tensorforge::intel_esimd::simd<float, 32> v363_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v363_data + (v335_data * v330_data));
              tensorforge::intel_esimd::simd<float, 32> v366_data;
              v366_data.copy_from(glb_m1 + (280_i32));
              float v367_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v369_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v369_data + (v366_data * v367_data));
              float v372_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v374_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v374_data + (v366_data * v372_data));
              float v377_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v379_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v379_data + (v366_data * v377_data));
              float v382_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v384_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v384_data + (v366_data * v382_data));
              float v387_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v389_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v389_data + (v366_data * v387_data));
              float v392_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v394_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v394_data + (v366_data * v392_data));
              tensorforge::intel_esimd::simd<float, 32> v397_data(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v397_data_part;
              v397_data_part.copy_from(glb_m1 + (312_i32));
              v397_data.template select<8, 1>(0) = v397_data_part;
              tensorforge::intel_esimd::simd<float, 32> v400_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v400_data + (v397_data * v367_data));
              tensorforge::intel_esimd::simd<float, 32> v405_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v405_data + (v397_data * v372_data));
              tensorforge::intel_esimd::simd<float, 32> v410_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v410_data + (v397_data * v377_data));
              tensorforge::intel_esimd::simd<float, 32> v415_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v415_data + (v397_data * v382_data));
              tensorforge::intel_esimd::simd<float, 32> v420_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v420_data + (v397_data * v387_data));
              tensorforge::intel_esimd::simd<float, 32> v425_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v425_data + (v397_data * v392_data));
              // r0 = ir0
              #pragma unroll
              for (int32_t v427_n0 = 0; v427_n0 < 1; ++v427_n0) {
                int32_t v429_a = v427_n0 * 32;
                #pragma unroll
                for (int32_t v428_n1 = 0; v428_n1 < 6; ++v428_n1) {
                  int32_t v431_a = v429_a + (v428_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v432_data(ir0.template select<32, 1>(v431_a));
                  r0.template select<32, 1>(v431_a) = v432_data;
                }
              }
              #pragma unroll
              for (int32_t v433_n1 = 0; v433_n1 < 6; ++v433_n1) {
                int32_t v435_a = 32 + (v433_n1 * 64);
                tensorforge::intel_esimd::simd<float, 8> v436_data(ir0.template select<8, 1>(v435_a));
                r0.template select<8, 1>(v435_a) = v436_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v437_i0 = 0; v437_i0 < 1; ++v437_i0) {
                int32_t v439_a = v437_i0 * 32;
                #pragma unroll
                for (int32_t v438_i1 = 0; v438_i1 < 6; ++v438_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v442_data(r0.template select<32, 1>((v439_a + (v438_i1 * 64))));
                  v442_data.copy_to(glb_m0 + ((v439_a + (v438_i1 * 40))));
                }
              }
              #pragma unroll
              for (int32_t v446_i1 = 0; v446_i1 < 6; ++v446_i1) {
                tensorforge::intel_esimd::simd<float, 8> v449_data(r0.template select<8, 1>((32 + (v446_i1 * 64))));
                v449_data.copy_to(glb_m0 + ((32_i32 + (v446_i1 * 40))));
              }
            }
            tensorforge::prefetchL2<48>(&pf_glb_m2[0]);
          }
        }
      }
    });
  });
}

