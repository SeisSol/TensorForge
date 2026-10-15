// === base name ===
kernel_cd5eca079e7f959c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_cd5eca079e7f959c = {{1, 32, 1}, 32, 24, 1, 32, 6848, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_cd5eca079e7f959c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_cd5eca079e7f959c(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_cd5eca079e7f959c(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 32, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 32 - 1) / 32;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 32;
  config.block[2] = 1;
  config.sharedMemBytes = 1712 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_cd5eca079e7f959c(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_cd5eca079e7f959c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_cd5eca079e7f959c(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_cd5eca079e7f959c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1712 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: pack_operands=bits
        // launch: 32 lanes (24 active) x 32 per block = block 1x32x1, 6848 B shared, occupancy grid
        // operands:
        //   m0 24×6(24×6) {0..24}×{0..6} strided
        //   m1 24×7(24×7) {0..24}×{0..7} none
        //   m2 7×6(7×6) {0..7}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":24,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":1712}],"shared_bytes":6848,"shared_elements":1712,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[24,6]],"name":"m0","ordered":false,"parts":1,"shape":[24,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[24,7]],"name":"m1","ordered":false,"parts":1,"shape":[24,7],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[7,6]],"name":"m2","ordered":false,"parts":1,"shape":[7,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[24,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[24,7]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,7]},{"addressing":"strided","bbox":[[0,0],[7,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[7,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (48 * item.get_local_id(1) + 176);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ ptr_glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> glb_m1 = totalShrMem + (0);
          // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 32> v9_ld;
            v9_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v9_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<float, 32> v10_ld;
            v10_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v10_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<float, 32> v11_ld;
            v11_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v11_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<float, 32> v12_ld;
            v12_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v12_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<float, 32> v13_ld;
            v13_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v13_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<float, 8> v14_ld;
            v14_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 8>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v14_ld);
          }
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v262_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v262_flagSecondAhead < numElements0) ? v262_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 42 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          tensorforge::intel_esimd::simd<float, 32> v270_ld;
          v270_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 0));
          tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v270_ld);
          tensorforge::intel_esimd::simd<float, 10> v271_ld;
          v271_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 32));
          tensorforge::slmStore<float, 10>(s0 + (0 + 0 + 1 * 0 + 32), v271_ld);
          uint32_t v266_flagWord(flagWordFirst);
          uint32_t v267_flagWordNext(flagWordSecond);
          for (size_t v16_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v16_batchId0 < numElements0; v16_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v17_ahead1 = v16_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v19_batchId1 = (v17_ahead1 < numElements0) ? v17_ahead1 : v16_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v19_batchId1 * 42 + 0 + m2_extraOffset];
            size_t v276_flagAheadAhead = v19_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v276_flagAheadAhead < numElements0) ? v276_flagAheadAhead : v19_batchId1)];
            const bool allowed = static_cast<bool>(v266_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v16_batchId0 * 144 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 24), (0, 6)] [(0, 7)]
              tensorforge::intel_esimd::simd<float, 192> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v33_data(0.0f);
              v33_data.template select<24, 1>(0) = tensorforge::slmLoad<float, 24>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 48> s0_w0 = tensorforge::slmLoad<float, 48>(s0 + 0);
              float v34_data = s0_w0[0];
              ir0.template select<32, 1>(0) = (v33_data * v34_data);
              float v39_data = s0_w0[7];
              ir0.template select<32, 1>(32) = (v33_data * v39_data);
              float v44_data = s0_w0[14];
              ir0.template select<32, 1>(64) = (v33_data * v44_data);
              float v49_data = s0_w0[21];
              ir0.template select<32, 1>(96) = (v33_data * v49_data);
              float v54_data = s0_w0[28];
              ir0.template select<32, 1>(128) = (v33_data * v54_data);
              float v59_data = s0_w0[35];
              ir0.template select<32, 1>(160) = (v33_data * v59_data);
              tensorforge::intel_esimd::simd<float, 32> v64_data(0.0f);
              v64_data.template select<24, 1>(0) = tensorforge::slmLoad<float, 24>(glb_m1 + (24_i32));
              float v65_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v67_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v67_data + (v64_data * v65_data));
              float v70_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v72_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v72_data + (v64_data * v70_data));
              float v75_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v77_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v77_data + (v64_data * v75_data));
              float v80_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v82_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v82_data + (v64_data * v80_data));
              float v85_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v87_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v87_data + (v64_data * v85_data));
              float v90_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v92_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v92_data + (v64_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v95_data(0.0f);
              v95_data.template select<24, 1>(0) = tensorforge::slmLoad<float, 24>(glb_m1 + (48_i32));
              float v96_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v98_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v98_data + (v95_data * v96_data));
              float v101_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v103_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v103_data + (v95_data * v101_data));
              float v106_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v108_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v108_data + (v95_data * v106_data));
              float v111_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v113_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v113_data + (v95_data * v111_data));
              float v116_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v118_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v118_data + (v95_data * v116_data));
              float v121_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v123_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v123_data + (v95_data * v121_data));
              tensorforge::intel_esimd::simd<float, 32> v126_data(0.0f);
              v126_data.template select<24, 1>(0) = tensorforge::slmLoad<float, 24>(glb_m1 + (72_i32));
              float v127_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v129_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v129_data + (v126_data * v127_data));
              float v132_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v134_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v134_data + (v126_data * v132_data));
              float v137_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v139_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v139_data + (v126_data * v137_data));
              float v142_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v144_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v144_data + (v126_data * v142_data));
              float v147_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v149_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v149_data + (v126_data * v147_data));
              float v152_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v154_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v154_data + (v126_data * v152_data));
              tensorforge::intel_esimd::simd<float, 32> v157_data(0.0f);
              v157_data.template select<24, 1>(0) = tensorforge::slmLoad<float, 24>(glb_m1 + (96_i32));
              float v158_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v160_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v160_data + (v157_data * v158_data));
              float v163_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v165_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v165_data + (v157_data * v163_data));
              float v168_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v170_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v170_data + (v157_data * v168_data));
              float v173_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v175_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v175_data + (v157_data * v173_data));
              float v178_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 32> v180_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v180_data + (v157_data * v178_data));
              float v183_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v185_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v185_data + (v157_data * v183_data));
              tensorforge::intel_esimd::simd<float, 32> v188_data(0.0f);
              v188_data.template select<24, 1>(0) = tensorforge::slmLoad<float, 24>(glb_m1 + (120_i32));
              float v189_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v191_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v191_data + (v188_data * v189_data));
              float v194_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v196_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v196_data + (v188_data * v194_data));
              float v199_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v201_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v201_data + (v188_data * v199_data));
              float v204_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v206_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v206_data + (v188_data * v204_data));
              float v209_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 32> v211_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v211_data + (v188_data * v209_data));
              float v214_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 32> v216_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v216_data + (v188_data * v214_data));
              tensorforge::intel_esimd::simd<float, 32> v219_data(0.0f);
              v219_data.template select<24, 1>(0) = tensorforge::slmLoad<float, 24>(glb_m1 + (144_i32));
              float v220_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v222_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v222_data + (v219_data * v220_data));
              float v225_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v227_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v227_data + (v219_data * v225_data));
              float v230_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v232_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v232_data + (v219_data * v230_data));
              float v235_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v237_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v237_data + (v219_data * v235_data));
              float v240_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v242_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v242_data + (v219_data * v240_data));
              float v245_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v247_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v247_data + (v219_data * v245_data));
              // r0 = ir0
              #pragma unroll
              for (int32_t v249_n1 = 0; v249_n1 < 6; ++v249_n1) {
                int32_t v250_a = v249_n1 * 32;
                tensorforge::intel_esimd::simd<float, 24> v252_data(ir0.template select<24, 1>(v250_a));
                r0.template select<24, 1>(v250_a) = v252_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v253_i1 = 0; v253_i1 < 6; ++v253_i1) {
                tensorforge::intel_esimd::simd<float, 24> v256_data(r0.template select<24, 1>((v253_i1 * 32)));
                v256_data.copy_to(glb_m0 + ((v253_i1 * 24)));
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 32> v272_ld;
            v272_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 0));
            tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v272_ld);
            tensorforge::intel_esimd::simd<float, 10> v273_ld;
            v273_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 32));
            tensorforge::slmStore<float, 10>(s0 + (0 + 0 + 1 * 0 + 32), v273_ld);
            v266_flagWord = v267_flagWordNext;
            v267_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}

