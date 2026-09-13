// === base name ===
kernel_c32cc7fa6477fada

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c32cc7fa6477fada = {{1, 32, 1}, 8, 8, 1, 32, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c32cc7fa6477fada(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c32cc7fa6477fada(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c32cc7fa6477fada(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 32, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 32;
  config.block[2] = 1;
  config.sharedMemBytes = 2304 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_c32cc7fa6477fada(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c32cc7fa6477fada(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_c32cc7fa6477fada(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_c32cc7fa6477fada(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 32 per block = block 1x32x1, 9216 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   C = abs(M)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"M","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (72 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (64);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const float *const __restrict__ pf_glb_m1 = &m1[v8_batchId1 * 64 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 64 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 64 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v5_batchId0 * 64 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 64> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v22_lead = v20_i0 * 8;
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 8; ++v21_i1) {
                  int32_t v25_a = v22_lead + (v21_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v26_data;
                  v26_data.copy_from(glb_m1 + (v25_a));
                  r0.template select<8, 1>(v25_a) = v26_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 0), v28_ld);
              tensorforge::intel_esimd::simd<float, 32> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 32));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 4 * 0 + 32), v29_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 64> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 64> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 8> v32_data(r0.template select<8, 1>(0));
              tensorforge::intel_esimd::simd<float, 64> s0_w0 = tensorforge::slmLoad<float, 64>(s0 + 0);
              float v33_data = s0_w0[0];
              tensorforge::intel_esimd::simd<float, 8> v35_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v35_data + (v32_data * v33_data));
              float v38_data = s0_w0[8];
              tensorforge::intel_esimd::simd<float, 8> v40_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v40_data + (v32_data * v38_data));
              float v43_data = s0_w0[16];
              tensorforge::intel_esimd::simd<float, 8> v45_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v45_data + (v32_data * v43_data));
              float v48_data = s0_w0[24];
              tensorforge::intel_esimd::simd<float, 8> v50_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v50_data + (v32_data * v48_data));
              float v53_data = s0_w0[32];
              tensorforge::intel_esimd::simd<float, 8> v55_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v55_data + (v32_data * v53_data));
              float v58_data = s0_w0[40];
              tensorforge::intel_esimd::simd<float, 8> v60_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v60_data + (v32_data * v58_data));
              float v63_data = s0_w0[48];
              tensorforge::intel_esimd::simd<float, 8> v65_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v65_data + (v32_data * v63_data));
              float v68_data = s0_w0[56];
              tensorforge::intel_esimd::simd<float, 8> v70_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v70_data + (v32_data * v68_data));
              tensorforge::intel_esimd::simd<float, 8> v72_data(r0.template select<8, 1>(8));
              float v73_data = s0_w0[1];
              tensorforge::intel_esimd::simd<float, 8> v75_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v75_data + (v72_data * v73_data));
              float v78_data = s0_w0[9];
              tensorforge::intel_esimd::simd<float, 8> v80_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v80_data + (v72_data * v78_data));
              float v83_data = s0_w0[17];
              tensorforge::intel_esimd::simd<float, 8> v85_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v85_data + (v72_data * v83_data));
              float v88_data = s0_w0[25];
              tensorforge::intel_esimd::simd<float, 8> v90_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v90_data + (v72_data * v88_data));
              float v93_data = s0_w0[33];
              tensorforge::intel_esimd::simd<float, 8> v95_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v95_data + (v72_data * v93_data));
              float v98_data = s0_w0[41];
              tensorforge::intel_esimd::simd<float, 8> v100_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v100_data + (v72_data * v98_data));
              float v103_data = s0_w0[49];
              tensorforge::intel_esimd::simd<float, 8> v105_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v105_data + (v72_data * v103_data));
              float v108_data = s0_w0[57];
              tensorforge::intel_esimd::simd<float, 8> v110_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v110_data + (v72_data * v108_data));
              tensorforge::intel_esimd::simd<float, 8> v112_data(r0.template select<8, 1>(16));
              float v113_data = s0_w0[2];
              tensorforge::intel_esimd::simd<float, 8> v115_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v115_data + (v112_data * v113_data));
              float v118_data = s0_w0[10];
              tensorforge::intel_esimd::simd<float, 8> v120_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v120_data + (v112_data * v118_data));
              float v123_data = s0_w0[18];
              tensorforge::intel_esimd::simd<float, 8> v125_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v125_data + (v112_data * v123_data));
              float v128_data = s0_w0[26];
              tensorforge::intel_esimd::simd<float, 8> v130_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v130_data + (v112_data * v128_data));
              float v133_data = s0_w0[34];
              tensorforge::intel_esimd::simd<float, 8> v135_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v135_data + (v112_data * v133_data));
              float v138_data = s0_w0[42];
              tensorforge::intel_esimd::simd<float, 8> v140_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v140_data + (v112_data * v138_data));
              float v143_data = s0_w0[50];
              tensorforge::intel_esimd::simd<float, 8> v145_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v145_data + (v112_data * v143_data));
              float v148_data = s0_w0[58];
              tensorforge::intel_esimd::simd<float, 8> v150_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v150_data + (v112_data * v148_data));
              tensorforge::intel_esimd::simd<float, 8> v152_data(r0.template select<8, 1>(24));
              float v153_data = s0_w0[3];
              tensorforge::intel_esimd::simd<float, 8> v155_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v155_data + (v152_data * v153_data));
              float v158_data = s0_w0[11];
              tensorforge::intel_esimd::simd<float, 8> v160_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v160_data + (v152_data * v158_data));
              float v163_data = s0_w0[19];
              tensorforge::intel_esimd::simd<float, 8> v165_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v165_data + (v152_data * v163_data));
              float v168_data = s0_w0[27];
              tensorforge::intel_esimd::simd<float, 8> v170_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v170_data + (v152_data * v168_data));
              float v173_data = s0_w0[35];
              tensorforge::intel_esimd::simd<float, 8> v175_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v175_data + (v152_data * v173_data));
              float v178_data = s0_w0[43];
              tensorforge::intel_esimd::simd<float, 8> v180_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v180_data + (v152_data * v178_data));
              float v183_data = s0_w0[51];
              tensorforge::intel_esimd::simd<float, 8> v185_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v185_data + (v152_data * v183_data));
              float v188_data = s0_w0[59];
              tensorforge::intel_esimd::simd<float, 8> v190_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v190_data + (v152_data * v188_data));
              tensorforge::intel_esimd::simd<float, 8> v192_data(r0.template select<8, 1>(32));
              float v193_data = s0_w0[4];
              tensorforge::intel_esimd::simd<float, 8> v195_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v195_data + (v192_data * v193_data));
              float v198_data = s0_w0[12];
              tensorforge::intel_esimd::simd<float, 8> v200_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v200_data + (v192_data * v198_data));
              float v203_data = s0_w0[20];
              tensorforge::intel_esimd::simd<float, 8> v205_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v205_data + (v192_data * v203_data));
              float v208_data = s0_w0[28];
              tensorforge::intel_esimd::simd<float, 8> v210_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v210_data + (v192_data * v208_data));
              float v213_data = s0_w0[36];
              tensorforge::intel_esimd::simd<float, 8> v215_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v215_data + (v192_data * v213_data));
              float v218_data = s0_w0[44];
              tensorforge::intel_esimd::simd<float, 8> v220_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v220_data + (v192_data * v218_data));
              float v223_data = s0_w0[52];
              tensorforge::intel_esimd::simd<float, 8> v225_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v225_data + (v192_data * v223_data));
              float v228_data = s0_w0[60];
              tensorforge::intel_esimd::simd<float, 8> v230_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v230_data + (v192_data * v228_data));
              tensorforge::intel_esimd::simd<float, 8> v232_data(r0.template select<8, 1>(40));
              float v233_data = s0_w0[5];
              tensorforge::intel_esimd::simd<float, 8> v235_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v235_data + (v232_data * v233_data));
              float v238_data = s0_w0[13];
              tensorforge::intel_esimd::simd<float, 8> v240_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v240_data + (v232_data * v238_data));
              float v243_data = s0_w0[21];
              tensorforge::intel_esimd::simd<float, 8> v245_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v245_data + (v232_data * v243_data));
              float v248_data = s0_w0[29];
              tensorforge::intel_esimd::simd<float, 8> v250_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v250_data + (v232_data * v248_data));
              float v253_data = s0_w0[37];
              tensorforge::intel_esimd::simd<float, 8> v255_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v255_data + (v232_data * v253_data));
              float v258_data = s0_w0[45];
              tensorforge::intel_esimd::simd<float, 8> v260_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v260_data + (v232_data * v258_data));
              float v263_data = s0_w0[53];
              tensorforge::intel_esimd::simd<float, 8> v265_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v265_data + (v232_data * v263_data));
              float v268_data = s0_w0[61];
              tensorforge::intel_esimd::simd<float, 8> v270_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v270_data + (v232_data * v268_data));
              tensorforge::intel_esimd::simd<float, 8> v272_data(r0.template select<8, 1>(48));
              float v273_data = s0_w0[6];
              tensorforge::intel_esimd::simd<float, 8> v275_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v275_data + (v272_data * v273_data));
              float v278_data = s0_w0[14];
              tensorforge::intel_esimd::simd<float, 8> v280_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v280_data + (v272_data * v278_data));
              float v283_data = s0_w0[22];
              tensorforge::intel_esimd::simd<float, 8> v285_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v285_data + (v272_data * v283_data));
              float v288_data = s0_w0[30];
              tensorforge::intel_esimd::simd<float, 8> v290_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v290_data + (v272_data * v288_data));
              float v293_data = s0_w0[38];
              tensorforge::intel_esimd::simd<float, 8> v295_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v295_data + (v272_data * v293_data));
              float v298_data = s0_w0[46];
              tensorforge::intel_esimd::simd<float, 8> v300_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v300_data + (v272_data * v298_data));
              float v303_data = s0_w0[54];
              tensorforge::intel_esimd::simd<float, 8> v305_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v305_data + (v272_data * v303_data));
              float v308_data = s0_w0[62];
              tensorforge::intel_esimd::simd<float, 8> v310_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v310_data + (v272_data * v308_data));
              tensorforge::intel_esimd::simd<float, 8> v312_data(r0.template select<8, 1>(56));
              float v313_data = s0_w0[7];
              tensorforge::intel_esimd::simd<float, 8> v315_data(ir1.template select<8, 1>(0));
              ir1.template select<8, 1>(0) = (v315_data + (v312_data * v313_data));
              float v318_data = s0_w0[15];
              tensorforge::intel_esimd::simd<float, 8> v320_data(ir1.template select<8, 1>(8));
              ir1.template select<8, 1>(8) = (v320_data + (v312_data * v318_data));
              float v323_data = s0_w0[23];
              tensorforge::intel_esimd::simd<float, 8> v325_data(ir1.template select<8, 1>(16));
              ir1.template select<8, 1>(16) = (v325_data + (v312_data * v323_data));
              float v328_data = s0_w0[31];
              tensorforge::intel_esimd::simd<float, 8> v330_data(ir1.template select<8, 1>(24));
              ir1.template select<8, 1>(24) = (v330_data + (v312_data * v328_data));
              float v333_data = s0_w0[39];
              tensorforge::intel_esimd::simd<float, 8> v335_data(ir1.template select<8, 1>(32));
              ir1.template select<8, 1>(32) = (v335_data + (v312_data * v333_data));
              float v338_data = s0_w0[47];
              tensorforge::intel_esimd::simd<float, 8> v340_data(ir1.template select<8, 1>(40));
              ir1.template select<8, 1>(40) = (v340_data + (v312_data * v338_data));
              float v343_data = s0_w0[55];
              tensorforge::intel_esimd::simd<float, 8> v345_data(ir1.template select<8, 1>(48));
              ir1.template select<8, 1>(48) = (v345_data + (v312_data * v343_data));
              float v348_data = s0_w0[63];
              tensorforge::intel_esimd::simd<float, 8> v350_data(ir1.template select<8, 1>(56));
              ir1.template select<8, 1>(56) = (v350_data + (v312_data * v348_data));
              // r1 = ir1
              #pragma unroll
              for (int32_t v352_n0 = 0; v352_n0 < 1; ++v352_n0) {
                int32_t v354_a = v352_n0 * 8;
                #pragma unroll
                for (int32_t v353_n1 = 0; v353_n1 < 8; ++v353_n1) {
                  int32_t v356_a = v354_a + (v353_n1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v357_data(ir1.template select<8, 1>(v356_a));
                  r1.template select<8, 1>(v356_a) = v357_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v358_i0 = 0; v358_i0 < 1; ++v358_i0) {
                int32_t v360_a = v358_i0 * 8;
                #pragma unroll
                for (int32_t v359_i1 = 0; v359_i1 < 8; ++v359_i1) {
                  int32_t v362_a = v360_a + (v359_i1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v363_data(r1.template select<8, 1>(v362_a));
                  v363_data.copy_to(glb_m0 + (v362_a));
                }
              }
              // glb_m3 = abs(glb_m0)
              #pragma unroll
              for (int32_t v366_k0 = 0; v366_k0 < 1; ++v366_k0) {
                int32_t v368_lead = v366_k0 * 8;
                #pragma unroll
                for (int32_t v367_k1 = 0; v367_k1 < 8; ++v367_k1) {
                  int32_t v371_a = v368_lead + (v367_k1 * 8);
                  tensorforge::intel_esimd::simd<float, 8> v372_data;
                  v372_data.copy_from(glb_m0 + (v371_a));
                  (tensorforge::intel_esimd::abs(v372_data)).copy_to(glb_m3 + (v371_a));
                }
              }
            }
            tensorforge::prefetchRunsL2<256, 256>(&pf_glb_m1[0], &pf_glb_m2[0]);
          }
        }
      }
    });
  });
}

