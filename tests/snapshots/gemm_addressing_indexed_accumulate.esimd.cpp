// === base name ===
kernel_a620ac78f762e09f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a620ac78f762e09f = {{1, 8, 1}, 32, 20, 1, 8, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a620ac78f762e09f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a620ac78f762e09f(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a620ac78f762e09f(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 8 - 1) / 8;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 768 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_a620ac78f762e09f(float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a620ac78f762e09f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a620ac78f762e09f(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
class kernel_kernel_a620ac78f762e09f_k;
inline void kernel_kernel_a620ac78f762e09f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float ** m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_kernel_a620ac78f762e09f_k>(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<768 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (20 active) x 8 per block = block 1x8x1, 3072 B shared, occupancy grid
        // operands:
        //   m0 20×9(20×9) {0..20}×{0..9} indexed_pointer(slots=2, slot=1)
        //   m1 20×9(20×9) {0..20}×{0..9} none
        //   m2 9×9(9×9) {0..9}×{0..9} indexed(offset=12)
        // operations:
        //   m0[i,j] += m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"indexed_pointer(slots=2, slot=1)","alias":"C","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[20,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"indexed(offset=12)","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"indexed_pointer(slots=2, slot=1)","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]},{"addressing":"indexed(offset=12)","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (96 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::intel_esimd::simd<float, 288> r0(0.0f);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v464_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v464_flagSecondAhead < numElements0) ? v464_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const size_t peel_elementIndex = static_cast<size_t>(elementIndices[v3_batchId1]);
          const float *const __restrict__ peel_glb_m2 = &m2[peel_elementIndex * 81 + 0 + 12 + m2_extraOffset];
          if (allowed_peel) {
            // s0 = load{g>s}(peel_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 64> v476_ld;
            v476_ld.copy_from(peel_glb_m2 + (0 + 0 + 2 * 0 + 0));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 2 * 0 + 0), v476_ld);
            tensorforge::intel_esimd::simd<float, 17> v477_ld;
            v477_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 64));
            tensorforge::slmStore<float, 17>(s0 + (0 + 0 + 1 * 0 + 64), v477_ld);
          }
          float *const __restrict__ peel_glb_m0 = &m0[peel_elementIndex * 2 + 1][0 + m0_extraOffset];
          if (allowed_peel) {
            // r0 = load{g>r}(peel_glb_m0);
            #pragma unroll
            for (int32_t v486_i1 = 0; v486_i1 < 9; ++v486_i1) {
              tensorforge::intel_esimd::simd<float, 32> v488_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v488_data_part;
              v488_data_part.copy_from(peel_glb_m0 + ((v486_i1 * 20)));
              v488_data.template select<20, 1>(0) = v488_data_part;
              r0.template select<32, 1>((v486_i1 * 32)) = v488_data;
            }
          }
          uint32_t v468_flagWord(flagWordFirst);
          uint32_t v469_flagWordNext(flagWordSecond);
          size_t v501_elementIndex(peel_elementIndex);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[v12_batchId1]);
            const float *const __restrict__ wrap_glb_m2 = &m2[wrap_elementIndex * 81 + 0 + 12 + m2_extraOffset];
            float *const __restrict__ wrap_glb_m0 = &m0[wrap_elementIndex * 2 + 1][0 + m0_extraOffset];
            size_t v496_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v496_flagAheadAhead < numElements0) ? v496_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v468_flagWord);
            const bool allowed_next = static_cast<bool>(v469_flagWordNext);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v501_elementIndex * 2 + 1][0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 288> r1(0.0f);
              // ir1 = +(glb_m1 * s0)
              // [(0, 20), (0, 9)] [(0, 9)]
              tensorforge::intel_esimd::simd<float, 288> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v27_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v27_data_part;
              v27_data_part.copy_from(glb_m1 + (0_i32));
              v27_data.template select<20, 1>(0) = v27_data_part;
              tensorforge::intel_esimd::simd<float, 96> s0_w0 = tensorforge::slmLoad<float, 96>(s0 + 0);
              float v28_data = s0_w0[0];
              float v33_data = s0_w0[9];
              float v38_data = s0_w0[18];
              float v43_data = s0_w0[27];
              float v48_data = s0_w0[36];
              float v53_data = s0_w0[45];
              float v58_data = s0_w0[54];
              float v63_data = s0_w0[63];
              float v68_data = s0_w0[72];
              tensorforge::intel_esimd::simd<float, 32> v73_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v73_data_part;
              v73_data_part.copy_from(glb_m1 + (20_i32));
              v73_data.template select<20, 1>(0) = v73_data_part;
              float v74_data = s0_w0[1];
              float v79_data = s0_w0[10];
              float v84_data = s0_w0[19];
              float v89_data = s0_w0[28];
              float v94_data = s0_w0[37];
              float v99_data = s0_w0[46];
              float v104_data = s0_w0[55];
              float v109_data = s0_w0[64];
              float v114_data = s0_w0[73];
              tensorforge::intel_esimd::simd<float, 32> v119_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v119_data_part;
              v119_data_part.copy_from(glb_m1 + (40_i32));
              v119_data.template select<20, 1>(0) = v119_data_part;
              float v120_data = s0_w0[2];
              float v125_data = s0_w0[11];
              float v130_data = s0_w0[20];
              float v135_data = s0_w0[29];
              float v140_data = s0_w0[38];
              float v145_data = s0_w0[47];
              float v150_data = s0_w0[56];
              float v155_data = s0_w0[65];
              float v160_data = s0_w0[74];
              tensorforge::intel_esimd::simd<float, 32> v165_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v165_data_part;
              v165_data_part.copy_from(glb_m1 + (60_i32));
              v165_data.template select<20, 1>(0) = v165_data_part;
              float v166_data = s0_w0[3];
              float v171_data = s0_w0[12];
              float v176_data = s0_w0[21];
              float v181_data = s0_w0[30];
              float v186_data = s0_w0[39];
              float v191_data = s0_w0[48];
              float v196_data = s0_w0[57];
              float v201_data = s0_w0[66];
              float v206_data = s0_w0[75];
              tensorforge::intel_esimd::simd<float, 32> v211_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v211_data_part;
              v211_data_part.copy_from(glb_m1 + (80_i32));
              v211_data.template select<20, 1>(0) = v211_data_part;
              float v212_data = s0_w0[4];
              float v217_data = s0_w0[13];
              float v222_data = s0_w0[22];
              float v227_data = s0_w0[31];
              float v232_data = s0_w0[40];
              float v237_data = s0_w0[49];
              float v242_data = s0_w0[58];
              float v247_data = s0_w0[67];
              float v252_data = s0_w0[76];
              tensorforge::intel_esimd::simd<float, 32> v257_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v257_data_part;
              v257_data_part.copy_from(glb_m1 + (100_i32));
              v257_data.template select<20, 1>(0) = v257_data_part;
              float v258_data = s0_w0[5];
              float v263_data = s0_w0[14];
              float v268_data = s0_w0[23];
              float v273_data = s0_w0[32];
              float v278_data = s0_w0[41];
              float v283_data = s0_w0[50];
              float v288_data = s0_w0[59];
              float v293_data = s0_w0[68];
              float v298_data = s0_w0[77];
              tensorforge::intel_esimd::simd<float, 32> v303_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v303_data_part;
              v303_data_part.copy_from(glb_m1 + (120_i32));
              v303_data.template select<20, 1>(0) = v303_data_part;
              float v304_data = s0_w0[6];
              float v309_data = s0_w0[15];
              float v314_data = s0_w0[24];
              float v319_data = s0_w0[33];
              float v324_data = s0_w0[42];
              float v329_data = s0_w0[51];
              float v334_data = s0_w0[60];
              float v339_data = s0_w0[69];
              float v344_data = s0_w0[78];
              tensorforge::intel_esimd::simd<float, 32> v349_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v349_data_part;
              v349_data_part.copy_from(glb_m1 + (140_i32));
              v349_data.template select<20, 1>(0) = v349_data_part;
              float v350_data = s0_w0[7];
              float v355_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v358_p = (((((((v27_data * v33_data) + (v73_data * v79_data)) + (v119_data * v125_data)) + (v165_data * v171_data)) + (v211_data * v217_data)) + (v257_data * v263_data)) + (v303_data * v309_data)) + (v349_data * v355_data);
              float v360_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v363_p = (((((((v27_data * v38_data) + (v73_data * v84_data)) + (v119_data * v130_data)) + (v165_data * v176_data)) + (v211_data * v222_data)) + (v257_data * v268_data)) + (v303_data * v314_data)) + (v349_data * v360_data);
              float v365_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v368_p = (((((((v27_data * v43_data) + (v73_data * v89_data)) + (v119_data * v135_data)) + (v165_data * v181_data)) + (v211_data * v227_data)) + (v257_data * v273_data)) + (v303_data * v319_data)) + (v349_data * v365_data);
              float v370_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v373_p = (((((((v27_data * v48_data) + (v73_data * v94_data)) + (v119_data * v140_data)) + (v165_data * v186_data)) + (v211_data * v232_data)) + (v257_data * v278_data)) + (v303_data * v324_data)) + (v349_data * v370_data);
              float v375_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 32> v378_p = (((((((v27_data * v53_data) + (v73_data * v99_data)) + (v119_data * v145_data)) + (v165_data * v191_data)) + (v211_data * v237_data)) + (v257_data * v283_data)) + (v303_data * v329_data)) + (v349_data * v375_data);
              float v380_data = s0_w0[61];
              tensorforge::intel_esimd::simd<float, 32> v383_p = (((((((v27_data * v58_data) + (v73_data * v104_data)) + (v119_data * v150_data)) + (v165_data * v196_data)) + (v211_data * v242_data)) + (v257_data * v288_data)) + (v303_data * v334_data)) + (v349_data * v380_data);
              float v385_data = s0_w0[70];
              tensorforge::intel_esimd::simd<float, 32> v388_p = (((((((v27_data * v63_data) + (v73_data * v109_data)) + (v119_data * v155_data)) + (v165_data * v201_data)) + (v211_data * v247_data)) + (v257_data * v293_data)) + (v303_data * v339_data)) + (v349_data * v385_data);
              float v390_data = s0_w0[79];
              tensorforge::intel_esimd::simd<float, 32> v393_p = (((((((v27_data * v68_data) + (v73_data * v114_data)) + (v119_data * v160_data)) + (v165_data * v206_data)) + (v211_data * v252_data)) + (v257_data * v298_data)) + (v303_data * v344_data)) + (v349_data * v390_data);
              tensorforge::intel_esimd::simd<float, 32> v395_data(0.0f);
              tensorforge::intel_esimd::simd<float, 20> v395_data_part;
              v395_data_part.copy_from(glb_m1 + (160_i32));
              v395_data.template select<20, 1>(0) = v395_data_part;
              float v396_data = s0_w0[8];
              ir1.template select<32, 1>(0) = (((((((((v27_data * v28_data) + (v73_data * v74_data)) + (v119_data * v120_data)) + (v165_data * v166_data)) + (v211_data * v212_data)) + (v257_data * v258_data)) + (v303_data * v304_data)) + (v349_data * v350_data)) + (v395_data * v396_data));
              float v401_data = s0_w0[17];
              ir1.template select<32, 1>(32) = (v358_p + (v395_data * v401_data));
              float v406_data = s0_w0[26];
              ir1.template select<32, 1>(64) = (v363_p + (v395_data * v406_data));
              float v411_data = s0_w0[35];
              ir1.template select<32, 1>(96) = (v368_p + (v395_data * v411_data));
              float v416_data = s0_w0[44];
              ir1.template select<32, 1>(128) = (v373_p + (v395_data * v416_data));
              float v421_data = s0_w0[53];
              ir1.template select<32, 1>(160) = (v378_p + (v395_data * v421_data));
              float v426_data = s0_w0[62];
              ir1.template select<32, 1>(192) = (v383_p + (v395_data * v426_data));
              float v431_data = s0_w0[71];
              ir1.template select<32, 1>(224) = (v388_p + (v395_data * v431_data));
              float v436_data = s0_w0[80];
              ir1.template select<32, 1>(256) = (v393_p + (v395_data * v436_data));
              // r1 = ir1 + r0
              #pragma unroll
              for (int32_t v449_n1 = 0; v449_n1 < 9; ++v449_n1) {
                int32_t v450_a = v449_n1 * 32;
                tensorforge::intel_esimd::simd<float, 20> v452_data(ir1.template select<20, 1>(v450_a));
                tensorforge::intel_esimd::simd<float, 20> v453_data(r0.template select<20, 1>(v450_a));
                r1.template select<20, 1>(v450_a) = (v453_data + v452_data);
              }
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m0);
                #pragma unroll
                for (int32_t v490_i1 = 0; v490_i1 < 9; ++v490_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v492_data(0.0f);
                  tensorforge::intel_esimd::simd<float, 20> v492_data_part;
                  v492_data_part.copy_from(wrap_glb_m0 + ((v490_i1 * 20)));
                  v492_data.template select<20, 1>(0) = v492_data_part;
                  r0.template select<32, 1>((v490_i1 * 32)) = v492_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v455_i1 = 0; v455_i1 < 9; ++v455_i1) {
                tensorforge::intel_esimd::simd<float, 20> v458_data(r1.template select<20, 1>((v455_i1 * 32)));
                v458_data.copy_to(glb_m0 + ((v455_i1 * 20)));
              }
            }
            else {
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m0);
                #pragma unroll
                for (int32_t v482_i1 = 0; v482_i1 < 9; ++v482_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v484_data(0.0f);
                  tensorforge::intel_esimd::simd<float, 20> v484_data_part;
                  v484_data_part.copy_from(wrap_glb_m0 + ((v482_i1 * 20)));
                  v484_data.template select<20, 1>(0) = v484_data_part;
                  r0.template select<32, 1>((v482_i1 * 32)) = v484_data;
                }
              }
            }
            if (allowed_next) {
              // s0 = load{g>s}(wrap_glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v478_ld;
              v478_ld.copy_from(wrap_glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 2 * 0 + 0), v478_ld);
              tensorforge::intel_esimd::simd<float, 17> v479_ld;
              v479_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<float, 17>(s0 + (0 + 0 + 1 * 0 + 64), v479_ld);
            }
            v468_flagWord = v469_flagWordNext;
            v469_flagWordNext = flagWordAhead;
            v501_elementIndex = wrap_elementIndex;
          }
        }
      }
    });
  });
}

