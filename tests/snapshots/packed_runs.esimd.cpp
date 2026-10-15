// === base name ===
kernel_bbb49c07057c0e69

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_bbb49c07057c0e69 = {{1, 32, 1}, 32, 40, 1, 32, 7424, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_bbb49c07057c0e69(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_bbb49c07057c0e69(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_bbb49c07057c0e69(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1856 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_bbb49c07057c0e69(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_bbb49c07057c0e69(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_bbb49c07057c0e69(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_bbb49c07057c0e69(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1856 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: pack_operands=runs
        // launch: 32 lanes (40 active) x 32 per block = block 1x32x1, 7424 B shared, occupancy grid
        // operands:
        //   m0 40×6(40×6) {0..40}×{0..6} strided
        //   m1 40×8(40×8) {0..40}×{0..8} none
        //   m2 8×6(8×6) {0..8}×{0..6} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":1856}],"shared_bytes":7424,"shared_elements":1856,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":false,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (48 * item.get_local_id(1) + 320);
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
            tensorforge::intel_esimd::simd<float, 32> v14_ld;
            v14_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v14_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<float, 32> v15_ld;
            v15_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v15_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<float, 32> v16_ld;
            v16_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v16_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<float, 32> v17_ld;
            v17_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v17_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<float, 32> v18_ld;
            v18_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 32>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v18_ld);
          }
          item.barrier();
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v376_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v376_flagSecondAhead < numElements0) ? v376_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          tensorforge::intel_esimd::simd<float, 32> v384_ld;
          v384_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 0));
          tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v384_ld);
          tensorforge::intel_esimd::simd<float, 16> v385_ld;
          v385_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 32));
          tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v385_ld);
          uint32_t v380_flagWord(flagWordFirst);
          uint32_t v381_flagWordNext(flagWordSecond);
          for (size_t v20_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v20_batchId0 < numElements0; v20_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v21_ahead1 = v20_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v23_batchId1 = (v21_ahead1 < numElements0) ? v21_ahead1 : v20_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v23_batchId1 * 48 + 0 + m2_extraOffset];
            size_t v390_flagAheadAhead = v23_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v390_flagAheadAhead < numElements0) ? v390_flagAheadAhead : v23_batchId1)];
            const bool allowed = static_cast<bool>(v380_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v20_batchId0 * 240 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 384> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 40), (0, 6)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 384> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v37_data = tensorforge::slmLoad<float, 32>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 48> s0_w0 = tensorforge::slmLoad<float, 48>(s0 + 0);
              float v38_data = s0_w0[0];
              ir0.template select<32, 1>(0) = (v37_data * v38_data);
              float v43_data = s0_w0[8];
              ir0.template select<32, 1>(64) = (v37_data * v43_data);
              float v48_data = s0_w0[16];
              ir0.template select<32, 1>(128) = (v37_data * v48_data);
              float v53_data = s0_w0[24];
              ir0.template select<32, 1>(192) = (v37_data * v53_data);
              float v58_data = s0_w0[32];
              ir0.template select<32, 1>(256) = (v37_data * v58_data);
              float v63_data = s0_w0[40];
              ir0.template select<32, 1>(320) = (v37_data * v63_data);
              tensorforge::intel_esimd::simd<float, 32> v68_data = tensorforge::slmLoad<float, 32>(glb_m1 + (40_i32));
              float v69_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v71_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v71_data + (v68_data * v69_data));
              float v74_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v76_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v76_data + (v68_data * v74_data));
              float v79_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v81_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v81_data + (v68_data * v79_data));
              float v84_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v86_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v86_data + (v68_data * v84_data));
              float v89_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 32> v91_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v91_data + (v68_data * v89_data));
              float v94_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v96_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v96_data + (v68_data * v94_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0 = tensorforge::slmLoad<float, 64>(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 32> v99_data(glb_m1_run0.template select<32, 1>(0));
              float v100_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v102_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v102_data + (v99_data * v100_data));
              float v105_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v107_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v107_data + (v99_data * v105_data));
              float v110_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v112_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v112_data + (v99_data * v110_data));
              float v115_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v117_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v117_data + (v99_data * v115_data));
              float v120_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v122_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v122_data + (v99_data * v120_data));
              float v125_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 32> v127_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v127_data + (v99_data * v125_data));
              tensorforge::intel_esimd::simd<float, 32> v131_data(glb_m1_run0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v131_data * v100_data);
              ir0.template select<32, 1>(96) = (v131_data * v105_data);
              ir0.template select<32, 1>(160) = (v131_data * v110_data);
              ir0.template select<32, 1>(224) = (v131_data * v115_data);
              ir0.template select<32, 1>(288) = (v131_data * v120_data);
              ir0.template select<32, 1>(352) = (v131_data * v125_data);
              tensorforge::intel_esimd::simd<float, 32> v162_data(0.0f);
              v162_data.template select<8, 1>(0) = tensorforge::slmLoad<float, 8>(glb_m1 + (152_i32));
              float v163_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v165_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v165_data + (v162_data * v163_data));
              float v168_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v170_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v170_data + (v162_data * v168_data));
              float v173_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v175_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v175_data + (v162_data * v173_data));
              float v178_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v180_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v180_data + (v162_data * v178_data));
              float v183_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v185_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v185_data + (v162_data * v183_data));
              float v188_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v190_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v190_data + (v162_data * v188_data));
              tensorforge::intel_esimd::simd<float, 32> v193_data = tensorforge::slmLoad<float, 32>(glb_m1 + (160_i32));
              float v194_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v196_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v196_data + (v193_data * v194_data));
              float v199_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v201_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v201_data + (v193_data * v199_data));
              float v204_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v206_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v206_data + (v193_data * v204_data));
              float v209_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v211_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v211_data + (v193_data * v209_data));
              float v214_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v216_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v216_data + (v193_data * v214_data));
              float v219_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v221_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v221_data + (v193_data * v219_data));
              tensorforge::intel_esimd::simd<float, 32> v224_data = tensorforge::slmLoad<float, 32>(glb_m1 + (200_i32));
              float v225_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v227_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v227_data + (v224_data * v225_data));
              float v230_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v232_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v232_data + (v224_data * v230_data));
              float v235_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v237_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v237_data + (v224_data * v235_data));
              float v240_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v242_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v242_data + (v224_data * v240_data));
              float v245_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v247_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v247_data + (v224_data * v245_data));
              float v250_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v252_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v252_data + (v224_data * v250_data));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1 = tensorforge::slmLoad<float, 64>(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 32> v255_data(glb_m1_run1.template select<32, 1>(0));
              float v256_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v258_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v258_data + (v255_data * v256_data));
              float v261_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v263_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v263_data + (v255_data * v261_data));
              float v266_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v268_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v268_data + (v255_data * v266_data));
              float v271_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v273_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v273_data + (v255_data * v271_data));
              float v276_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v278_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v278_data + (v255_data * v276_data));
              float v281_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v283_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v283_data + (v255_data * v281_data));
              tensorforge::intel_esimd::simd<float, 32> v286_data(glb_m1_run1.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v289_data(ir0.template select<32, 1>(32));
              ir0.template select<32, 1>(32) = (v289_data + (v286_data * v256_data));
              tensorforge::intel_esimd::simd<float, 32> v294_data(ir0.template select<32, 1>(96));
              ir0.template select<32, 1>(96) = (v294_data + (v286_data * v261_data));
              tensorforge::intel_esimd::simd<float, 32> v299_data(ir0.template select<32, 1>(160));
              ir0.template select<32, 1>(160) = (v299_data + (v286_data * v266_data));
              tensorforge::intel_esimd::simd<float, 32> v304_data(ir0.template select<32, 1>(224));
              ir0.template select<32, 1>(224) = (v304_data + (v286_data * v271_data));
              tensorforge::intel_esimd::simd<float, 32> v309_data(ir0.template select<32, 1>(288));
              ir0.template select<32, 1>(288) = (v309_data + (v286_data * v276_data));
              tensorforge::intel_esimd::simd<float, 32> v314_data(ir0.template select<32, 1>(352));
              ir0.template select<32, 1>(352) = (v314_data + (v286_data * v281_data));
              tensorforge::intel_esimd::simd<float, 32> v317_data = tensorforge::slmLoad<float, 32>(glb_m1 + (280_i32));
              float v318_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v320_data(ir0.template select<32, 1>(0));
              ir0.template select<32, 1>(0) = (v320_data + (v317_data * v318_data));
              float v323_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v325_data(ir0.template select<32, 1>(64));
              ir0.template select<32, 1>(64) = (v325_data + (v317_data * v323_data));
              float v328_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v330_data(ir0.template select<32, 1>(128));
              ir0.template select<32, 1>(128) = (v330_data + (v317_data * v328_data));
              float v333_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v335_data(ir0.template select<32, 1>(192));
              ir0.template select<32, 1>(192) = (v335_data + (v317_data * v333_data));
              float v338_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v340_data(ir0.template select<32, 1>(256));
              ir0.template select<32, 1>(256) = (v340_data + (v317_data * v338_data));
              float v343_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v345_data(ir0.template select<32, 1>(320));
              ir0.template select<32, 1>(320) = (v345_data + (v317_data * v343_data));
              // r0 = ir0
              #pragma unroll
              for (int32_t v347_n0 = 0; v347_n0 < 1; ++v347_n0) {
                int32_t v349_a = v347_n0 * 32;
                #pragma unroll
                for (int32_t v348_n1 = 0; v348_n1 < 6; ++v348_n1) {
                  int32_t v351_a = v349_a + (v348_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v352_data(ir0.template select<32, 1>(v351_a));
                  r0.template select<32, 1>(v351_a) = v352_data;
                }
              }
              #pragma unroll
              for (int32_t v353_n1 = 0; v353_n1 < 6; ++v353_n1) {
                int32_t v355_a = 32 + (v353_n1 * 64);
                tensorforge::intel_esimd::simd<float, 8> v356_data(ir0.template select<8, 1>(v355_a));
                r0.template select<8, 1>(v355_a) = v356_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v357_i0 = 0; v357_i0 < 1; ++v357_i0) {
                int32_t v359_a = v357_i0 * 32;
                #pragma unroll
                for (int32_t v358_i1 = 0; v358_i1 < 6; ++v358_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v362_data(r0.template select<32, 1>((v359_a + (v358_i1 * 64))));
                  v362_data.copy_to(glb_m0 + ((v359_a + (v358_i1 * 40))));
                }
              }
              #pragma unroll
              for (int32_t v366_i1 = 0; v366_i1 < 6; ++v366_i1) {
                tensorforge::intel_esimd::simd<float, 8> v369_data(r0.template select<8, 1>((32 + (v366_i1 * 64))));
                v369_data.copy_to(glb_m0 + ((32_i32 + (v366_i1 * 40))));
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 32> v386_ld;
            v386_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 0));
            tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 1 * 0 + 0), v386_ld);
            tensorforge::intel_esimd::simd<float, 16> v387_ld;
            v387_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 32));
            tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 32), v387_ld);
            v380_flagWord = v381_flagWordNext;
            v381_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}

