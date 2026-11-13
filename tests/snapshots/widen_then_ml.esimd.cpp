// === base name ===
kernel_2fc98e0df61f21bf

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_2fc98e0df61f21bf = {{1, 8, 1}, 32, 20, 1, 8, 6144, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_2fc98e0df61f21bf(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_2fc98e0df61f21bf(const float * m0, size_t m0_extraOffset, double * m1, size_t m1_extraOffset, const double * m2, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_2fc98e0df61f21bf(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1536 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_2fc98e0df61f21bf(const float * m0, size_t m0_extraOffset, double * m1, size_t m1_extraOffset, const double * m2, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_2fc98e0df61f21bf(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_2fc98e0df61f21bf(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
class kernel_kernel_2fc98e0df61f21bf_k;
inline void kernel_kernel_2fc98e0df61f21bf(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, double * m1, size_t m1_extraOffset, const double * m2, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_kernel_2fc98e0df61f21bf_k>(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1536 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (20 active) x 8 per block = block 1x8x1, 6144 B shared, occupancy grid
        // operands:
        //   m0 10×9(10×9) {0..10}×{0..9} strided
        //   m1 20×9(20×9) {0..20}×{0..9} strided
        //   m2 20×10(20×10) {0..20}×{0..10} none
        // operations:
        //   wide = copy(I)
        //   m1[i,j] = m2[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1536}],"shared_bytes":6144,"shared_elements":1536,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"I","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"strided","alias":"canonical","bbox":[[0,0],[20,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"embed","bbox":[[0,0],[20,10]],"name":"m2","ordered":false,"parts":1,"shape":[20,10],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"elementwise","op":"COPY","ops":[{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,10]},{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (192 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const double *const __restrict__ glb_m2 = &m2[0];
          tensorforge::SlmPtr<double> s0 = tensorforge::slmCast<double>(localShrMem0 + (0));
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 90 + 0 + m0_extraOffset];
              double *const __restrict__ glb_m1 = &m1[v9_batchId0 * 180 + 0 + m1_extraOffset];
              tensorforge::intel_esimd::simd<double, 288> r0(0.0);
              // r0 = copy(glb_m0)
              #pragma unroll
              for (int32_t v20_k1 = 0; v20_k1 < 9; ++v20_k1) {
                tensorforge::intel_esimd::simd<float, 10> v25_data;
                v25_data.copy_from(glb_m0 + ((v20_k1 * 10)));
                r0.template select<10, 1>((v20_k1 * 32)) = v25_data;
              }
              // s0 = store{r>s}(localShrMem0, r0);
              #pragma unroll
              for (int32_t v29_i1 = 0; v29_i1 < 9; ++v29_i1) {
                tensorforge::intel_esimd::simd<double, 10> v32_data(r0.template select<10, 1>((v29_i1 * 32)));
                tensorforge::slmStore<double, 10>(s0 + ((v29_i1 * 10)), v32_data);
              }
              tensorforge::intel_esimd::simd<double, 288> r1(0.0);
              // ir1 = +(glb_m2 * s0)
              // [(0, 20), (0, 9)] [(0, 10)]
              tensorforge::intel_esimd::simd<double, 288> ir1(0.0);
              tensorforge::intel_esimd::simd<double, 32> v42_data(0.0);
              tensorforge::intel_esimd::simd<double, 20> v42_data_part;
              v42_data_part.copy_from(glb_m2 + (0_i32));
              v42_data.template select<20, 1>(0) = v42_data_part;
              tensorforge::intel_esimd::simd<double, 96> s0_w0 = tensorforge::slmLoad<double, 96>(s0 + 0);
              double v43_data = s0_w0[0];
              double v48_data = s0_w0[10];
              double v53_data = s0_w0[20];
              double v58_data = s0_w0[30];
              double v63_data = s0_w0[40];
              double v68_data = s0_w0[50];
              double v73_data = s0_w0[60];
              double v78_data = s0_w0[70];
              double v83_data = s0_w0[80];
              tensorforge::intel_esimd::simd<double, 32> v88_data(0.0);
              tensorforge::intel_esimd::simd<double, 20> v88_data_part;
              v88_data_part.copy_from(glb_m2 + (20_i32));
              v88_data.template select<20, 1>(0) = v88_data_part;
              double v89_data = s0_w0[1];
              double v94_data = s0_w0[11];
              double v99_data = s0_w0[21];
              double v104_data = s0_w0[31];
              double v109_data = s0_w0[41];
              double v114_data = s0_w0[51];
              double v119_data = s0_w0[61];
              double v124_data = s0_w0[71];
              double v129_data = s0_w0[81];
              tensorforge::intel_esimd::simd<double, 32> v134_data(0.0);
              tensorforge::intel_esimd::simd<double, 20> v134_data_part;
              v134_data_part.copy_from(glb_m2 + (40_i32));
              v134_data.template select<20, 1>(0) = v134_data_part;
              double v135_data = s0_w0[2];
              double v140_data = s0_w0[12];
              double v145_data = s0_w0[22];
              double v150_data = s0_w0[32];
              double v155_data = s0_w0[42];
              double v160_data = s0_w0[52];
              double v165_data = s0_w0[62];
              double v170_data = s0_w0[72];
              double v175_data = s0_w0[82];
              tensorforge::intel_esimd::simd<double, 32> v180_data(0.0);
              tensorforge::intel_esimd::simd<double, 20> v180_data_part;
              v180_data_part.copy_from(glb_m2 + (60_i32));
              v180_data.template select<20, 1>(0) = v180_data_part;
              double v181_data = s0_w0[3];
              double v186_data = s0_w0[13];
              double v191_data = s0_w0[23];
              double v196_data = s0_w0[33];
              double v201_data = s0_w0[43];
              double v206_data = s0_w0[53];
              double v211_data = s0_w0[63];
              double v216_data = s0_w0[73];
              double v221_data = s0_w0[83];
              tensorforge::intel_esimd::simd<double, 32> v226_data(0.0);
              tensorforge::intel_esimd::simd<double, 20> v226_data_part;
              v226_data_part.copy_from(glb_m2 + (80_i32));
              v226_data.template select<20, 1>(0) = v226_data_part;
              double v227_data = s0_w0[4];
              double v232_data = s0_w0[14];
              double v237_data = s0_w0[24];
              double v242_data = s0_w0[34];
              double v247_data = s0_w0[44];
              double v252_data = s0_w0[54];
              double v257_data = s0_w0[64];
              double v262_data = s0_w0[74];
              double v267_data = s0_w0[84];
              tensorforge::intel_esimd::simd<double, 32> v272_data(0.0);
              tensorforge::intel_esimd::simd<double, 20> v272_data_part;
              v272_data_part.copy_from(glb_m2 + (100_i32));
              v272_data.template select<20, 1>(0) = v272_data_part;
              double v273_data = s0_w0[5];
              double v278_data = s0_w0[15];
              double v283_data = s0_w0[25];
              double v288_data = s0_w0[35];
              double v293_data = s0_w0[45];
              double v298_data = s0_w0[55];
              double v303_data = s0_w0[65];
              double v308_data = s0_w0[75];
              double v313_data = s0_w0[85];
              tensorforge::intel_esimd::simd<double, 32> v318_data(0.0);
              tensorforge::intel_esimd::simd<double, 20> v318_data_part;
              v318_data_part.copy_from(glb_m2 + (120_i32));
              v318_data.template select<20, 1>(0) = v318_data_part;
              double v319_data = s0_w0[6];
              double v324_data = s0_w0[16];
              double v329_data = s0_w0[26];
              double v334_data = s0_w0[36];
              double v339_data = s0_w0[46];
              double v344_data = s0_w0[56];
              double v349_data = s0_w0[66];
              double v354_data = s0_w0[76];
              double v359_data = s0_w0[86];
              tensorforge::intel_esimd::simd<double, 32> v364_data(0.0);
              tensorforge::intel_esimd::simd<double, 20> v364_data_part;
              v364_data_part.copy_from(glb_m2 + (140_i32));
              v364_data.template select<20, 1>(0) = v364_data_part;
              double v365_data = s0_w0[7];
              double v370_data = s0_w0[17];
              double v375_data = s0_w0[27];
              double v380_data = s0_w0[37];
              double v385_data = s0_w0[47];
              double v390_data = s0_w0[57];
              double v395_data = s0_w0[67];
              double v400_data = s0_w0[77];
              double v405_data = s0_w0[87];
              tensorforge::intel_esimd::simd<double, 32> v410_data(0.0);
              tensorforge::intel_esimd::simd<double, 20> v410_data_part;
              v410_data_part.copy_from(glb_m2 + (160_i32));
              v410_data.template select<20, 1>(0) = v410_data_part;
              double v411_data = s0_w0[8];
              double v416_data = s0_w0[18];
              tensorforge::intel_esimd::simd<double, 32> v419_p = ((((((((v42_data * v48_data) + (v88_data * v94_data)) + (v134_data * v140_data)) + (v180_data * v186_data)) + (v226_data * v232_data)) + (v272_data * v278_data)) + (v318_data * v324_data)) + (v364_data * v370_data)) + (v410_data * v416_data);
              double v421_data = s0_w0[28];
              tensorforge::intel_esimd::simd<double, 32> v424_p = ((((((((v42_data * v53_data) + (v88_data * v99_data)) + (v134_data * v145_data)) + (v180_data * v191_data)) + (v226_data * v237_data)) + (v272_data * v283_data)) + (v318_data * v329_data)) + (v364_data * v375_data)) + (v410_data * v421_data);
              double v426_data = s0_w0[38];
              tensorforge::intel_esimd::simd<double, 32> v429_p = ((((((((v42_data * v58_data) + (v88_data * v104_data)) + (v134_data * v150_data)) + (v180_data * v196_data)) + (v226_data * v242_data)) + (v272_data * v288_data)) + (v318_data * v334_data)) + (v364_data * v380_data)) + (v410_data * v426_data);
              double v431_data = s0_w0[48];
              tensorforge::intel_esimd::simd<double, 32> v434_p = ((((((((v42_data * v63_data) + (v88_data * v109_data)) + (v134_data * v155_data)) + (v180_data * v201_data)) + (v226_data * v247_data)) + (v272_data * v293_data)) + (v318_data * v339_data)) + (v364_data * v385_data)) + (v410_data * v431_data);
              double v436_data = s0_w0[58];
              tensorforge::intel_esimd::simd<double, 32> v439_p = ((((((((v42_data * v68_data) + (v88_data * v114_data)) + (v134_data * v160_data)) + (v180_data * v206_data)) + (v226_data * v252_data)) + (v272_data * v298_data)) + (v318_data * v344_data)) + (v364_data * v390_data)) + (v410_data * v436_data);
              double v441_data = s0_w0[68];
              tensorforge::intel_esimd::simd<double, 32> v444_p = ((((((((v42_data * v73_data) + (v88_data * v119_data)) + (v134_data * v165_data)) + (v180_data * v211_data)) + (v226_data * v257_data)) + (v272_data * v303_data)) + (v318_data * v349_data)) + (v364_data * v395_data)) + (v410_data * v441_data);
              double v446_data = s0_w0[78];
              tensorforge::intel_esimd::simd<double, 32> v449_p = ((((((((v42_data * v78_data) + (v88_data * v124_data)) + (v134_data * v170_data)) + (v180_data * v216_data)) + (v226_data * v262_data)) + (v272_data * v308_data)) + (v318_data * v354_data)) + (v364_data * v400_data)) + (v410_data * v446_data);
              double v451_data = s0_w0[88];
              tensorforge::intel_esimd::simd<double, 32> v454_p = ((((((((v42_data * v83_data) + (v88_data * v129_data)) + (v134_data * v175_data)) + (v180_data * v221_data)) + (v226_data * v267_data)) + (v272_data * v313_data)) + (v318_data * v359_data)) + (v364_data * v405_data)) + (v410_data * v451_data);
              tensorforge::intel_esimd::simd<double, 32> v456_data(0.0);
              tensorforge::intel_esimd::simd<double, 20> v456_data_part;
              v456_data_part.copy_from(glb_m2 + (180_i32));
              v456_data.template select<20, 1>(0) = v456_data_part;
              double v457_data = s0_w0[9];
              ir1.template select<32, 1>(0) = ((((((((((v42_data * v43_data) + (v88_data * v89_data)) + (v134_data * v135_data)) + (v180_data * v181_data)) + (v226_data * v227_data)) + (v272_data * v273_data)) + (v318_data * v319_data)) + (v364_data * v365_data)) + (v410_data * v411_data)) + (v456_data * v457_data));
              double v462_data = s0_w0[19];
              ir1.template select<32, 1>(32) = (v419_p + (v456_data * v462_data));
              double v467_data = s0_w0[29];
              ir1.template select<32, 1>(64) = (v424_p + (v456_data * v467_data));
              double v472_data = s0_w0[39];
              ir1.template select<32, 1>(96) = (v429_p + (v456_data * v472_data));
              double v477_data = s0_w0[49];
              ir1.template select<32, 1>(128) = (v434_p + (v456_data * v477_data));
              double v482_data = s0_w0[59];
              ir1.template select<32, 1>(160) = (v439_p + (v456_data * v482_data));
              double v487_data = s0_w0[69];
              ir1.template select<32, 1>(192) = (v444_p + (v456_data * v487_data));
              double v492_data = s0_w0[79];
              ir1.template select<32, 1>(224) = (v449_p + (v456_data * v492_data));
              double v497_data = s0_w0[89];
              ir1.template select<32, 1>(256) = (v454_p + (v456_data * v497_data));
              // r1 = ir1
              // glb_m1 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v505_i1 = 0; v505_i1 < 9; ++v505_i1) {
                tensorforge::intel_esimd::simd<double, 20> v508_data(ir1.template select<20, 1>((v505_i1 * 32)));
                v508_data.copy_to(glb_m1 + ((v505_i1 * 20)));
              }
            }
          }
        }
      }
    });
  });
}

