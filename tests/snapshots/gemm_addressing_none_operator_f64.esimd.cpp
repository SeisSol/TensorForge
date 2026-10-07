// === base name ===
kernel_68ef8eef0b13b1f8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_68ef8eef0b13b1f8 = {{1, 32, 1}, 16, 16, 1, 32, 71680, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_68ef8eef0b13b1f8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_68ef8eef0b13b1f8(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_68ef8eef0b13b1f8(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 8960 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_68ef8eef0b13b1f8(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_68ef8eef0b13b1f8(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_68ef8eef0b13b1f8(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_68ef8eef0b13b1f8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<8960 * sizeof(double)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 32 per block = block 1x32x1, 71680 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} none
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":8960}],"shared_bytes":71680,"shared_elements":8960,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<double> totalShrMem = tensorforge::SlmPtr<double>(0);
          tensorforge::SlmPtr<double> localShrMem0 = totalShrMem + (272 * item.get_local_id(1) + 256);
          const double *const __restrict__ ptr_glb_m1 = &m1[0];
          tensorforge::SlmPtr<double> glb_m1 = totalShrMem + (0);
          // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<double, 16> v9_ld;
            v9_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v9_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<double, 16> v10_ld;
            v10_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v10_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<double, 16> v11_ld;
            v11_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v11_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<double, 16> v12_ld;
            v12_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v12_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<double, 16> v13_ld;
            v13_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v13_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<double, 16> v14_ld;
            v14_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v14_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<double, 16> v15_ld;
            v15_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v15_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<double, 16> v16_ld;
            v16_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v16_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<double, 16> v17_ld;
            v17_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v17_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<double, 16> v18_ld;
            v18_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v18_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<double, 16> v19_ld;
            v19_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v19_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<double, 16> v20_ld;
            v20_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v20_ld);
          }
          if (item.get_local_id(1) == 12) {
            tensorforge::intel_esimd::simd<double, 16> v21_ld;
            v21_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v21_ld);
          }
          if (item.get_local_id(1) == 13) {
            tensorforge::intel_esimd::simd<double, 16> v22_ld;
            v22_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v22_ld);
          }
          if (item.get_local_id(1) == 14) {
            tensorforge::intel_esimd::simd<double, 16> v23_ld;
            v23_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v23_ld);
          }
          if (item.get_local_id(1) == 15) {
            tensorforge::intel_esimd::simd<double, 16> v24_ld;
            v24_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v24_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<double> s0 = localShrMem0 + (0);
          for (size_t v26_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v26_batchId0 < numElements0; v26_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v27_ahead1 = v26_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v29_batchId1 = (v27_ahead1 < numElements0) ? v27_ahead1 : v26_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v26_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v26_batchId0 * 256 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v26_batchId0 * 256 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              #pragma unroll
              for (int32_t i = 0; i < 16; i += 2) {
                tensorforge::intel_esimd::simd<double, 32> v36_ld;
                v36_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + i * 16));
                tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + i * 16), v36_ld);
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<double, 256> r0(0.0);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<double, 256> ir0(0.0);
              tensorforge::intel_esimd::simd<double, 16> v42_data = tensorforge::slmLoad<double, 16>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<double, 256> s0_w0 = tensorforge::slmLoad<double, 256>(s0 + 0);
              double v43_data = s0_w0[0];
              tensorforge::intel_esimd::simd<double, 16> v45_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v45_data + (v42_data * v43_data));
              double v48_data = s0_w0[16];
              tensorforge::intel_esimd::simd<double, 16> v50_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v50_data + (v42_data * v48_data));
              double v53_data = s0_w0[32];
              tensorforge::intel_esimd::simd<double, 16> v55_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v55_data + (v42_data * v53_data));
              double v58_data = s0_w0[48];
              tensorforge::intel_esimd::simd<double, 16> v60_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v60_data + (v42_data * v58_data));
              double v63_data = s0_w0[64];
              tensorforge::intel_esimd::simd<double, 16> v65_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v65_data + (v42_data * v63_data));
              double v68_data = s0_w0[80];
              tensorforge::intel_esimd::simd<double, 16> v70_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v70_data + (v42_data * v68_data));
              double v73_data = s0_w0[96];
              tensorforge::intel_esimd::simd<double, 16> v75_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v75_data + (v42_data * v73_data));
              double v78_data = s0_w0[112];
              tensorforge::intel_esimd::simd<double, 16> v80_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v80_data + (v42_data * v78_data));
              double v83_data = s0_w0[128];
              tensorforge::intel_esimd::simd<double, 16> v85_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v85_data + (v42_data * v83_data));
              double v88_data = s0_w0[144];
              tensorforge::intel_esimd::simd<double, 16> v90_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v90_data + (v42_data * v88_data));
              double v93_data = s0_w0[160];
              tensorforge::intel_esimd::simd<double, 16> v95_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v95_data + (v42_data * v93_data));
              double v98_data = s0_w0[176];
              tensorforge::intel_esimd::simd<double, 16> v100_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v100_data + (v42_data * v98_data));
              double v103_data = s0_w0[192];
              tensorforge::intel_esimd::simd<double, 16> v105_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v105_data + (v42_data * v103_data));
              double v108_data = s0_w0[208];
              tensorforge::intel_esimd::simd<double, 16> v110_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v110_data + (v42_data * v108_data));
              double v113_data = s0_w0[224];
              tensorforge::intel_esimd::simd<double, 16> v115_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v115_data + (v42_data * v113_data));
              double v118_data = s0_w0[240];
              tensorforge::intel_esimd::simd<double, 16> v120_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v120_data + (v42_data * v118_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run0 = tensorforge::slmLoad<double, 32>(glb_m1 + (16_i32));
              tensorforge::intel_esimd::simd<double, 16> v123_data(glb_m1_run0.template select<16, 1>(0));
              double v124_data = s0_w0[1];
              tensorforge::intel_esimd::simd<double, 16> v126_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v126_data + (v123_data * v124_data));
              double v129_data = s0_w0[17];
              tensorforge::intel_esimd::simd<double, 16> v131_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v131_data + (v123_data * v129_data));
              double v134_data = s0_w0[33];
              tensorforge::intel_esimd::simd<double, 16> v136_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v136_data + (v123_data * v134_data));
              double v139_data = s0_w0[49];
              tensorforge::intel_esimd::simd<double, 16> v141_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v141_data + (v123_data * v139_data));
              double v144_data = s0_w0[65];
              tensorforge::intel_esimd::simd<double, 16> v146_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v146_data + (v123_data * v144_data));
              double v149_data = s0_w0[81];
              tensorforge::intel_esimd::simd<double, 16> v151_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v151_data + (v123_data * v149_data));
              double v154_data = s0_w0[97];
              tensorforge::intel_esimd::simd<double, 16> v156_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v156_data + (v123_data * v154_data));
              double v159_data = s0_w0[113];
              tensorforge::intel_esimd::simd<double, 16> v161_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v161_data + (v123_data * v159_data));
              double v164_data = s0_w0[129];
              tensorforge::intel_esimd::simd<double, 16> v166_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v166_data + (v123_data * v164_data));
              double v169_data = s0_w0[145];
              tensorforge::intel_esimd::simd<double, 16> v171_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v171_data + (v123_data * v169_data));
              double v174_data = s0_w0[161];
              tensorforge::intel_esimd::simd<double, 16> v176_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v176_data + (v123_data * v174_data));
              double v179_data = s0_w0[177];
              tensorforge::intel_esimd::simd<double, 16> v181_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v181_data + (v123_data * v179_data));
              double v184_data = s0_w0[193];
              tensorforge::intel_esimd::simd<double, 16> v186_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v186_data + (v123_data * v184_data));
              double v189_data = s0_w0[209];
              tensorforge::intel_esimd::simd<double, 16> v191_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v191_data + (v123_data * v189_data));
              double v194_data = s0_w0[225];
              tensorforge::intel_esimd::simd<double, 16> v196_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v196_data + (v123_data * v194_data));
              double v199_data = s0_w0[241];
              tensorforge::intel_esimd::simd<double, 16> v201_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v201_data + (v123_data * v199_data));
              tensorforge::intel_esimd::simd<double, 16> v204_data(glb_m1_run0.template select<16, 1>(16));
              double v205_data = s0_w0[2];
              tensorforge::intel_esimd::simd<double, 16> v207_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v207_data + (v204_data * v205_data));
              double v210_data = s0_w0[18];
              tensorforge::intel_esimd::simd<double, 16> v212_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v212_data + (v204_data * v210_data));
              double v215_data = s0_w0[34];
              tensorforge::intel_esimd::simd<double, 16> v217_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v217_data + (v204_data * v215_data));
              double v220_data = s0_w0[50];
              tensorforge::intel_esimd::simd<double, 16> v222_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v222_data + (v204_data * v220_data));
              double v225_data = s0_w0[66];
              tensorforge::intel_esimd::simd<double, 16> v227_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v227_data + (v204_data * v225_data));
              double v230_data = s0_w0[82];
              tensorforge::intel_esimd::simd<double, 16> v232_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v232_data + (v204_data * v230_data));
              double v235_data = s0_w0[98];
              tensorforge::intel_esimd::simd<double, 16> v237_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v237_data + (v204_data * v235_data));
              double v240_data = s0_w0[114];
              tensorforge::intel_esimd::simd<double, 16> v242_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v242_data + (v204_data * v240_data));
              double v245_data = s0_w0[130];
              tensorforge::intel_esimd::simd<double, 16> v247_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v247_data + (v204_data * v245_data));
              double v250_data = s0_w0[146];
              tensorforge::intel_esimd::simd<double, 16> v252_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v252_data + (v204_data * v250_data));
              double v255_data = s0_w0[162];
              tensorforge::intel_esimd::simd<double, 16> v257_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v257_data + (v204_data * v255_data));
              double v260_data = s0_w0[178];
              tensorforge::intel_esimd::simd<double, 16> v262_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v262_data + (v204_data * v260_data));
              double v265_data = s0_w0[194];
              tensorforge::intel_esimd::simd<double, 16> v267_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v267_data + (v204_data * v265_data));
              double v270_data = s0_w0[210];
              tensorforge::intel_esimd::simd<double, 16> v272_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v272_data + (v204_data * v270_data));
              double v275_data = s0_w0[226];
              tensorforge::intel_esimd::simd<double, 16> v277_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v277_data + (v204_data * v275_data));
              double v280_data = s0_w0[242];
              tensorforge::intel_esimd::simd<double, 16> v282_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v282_data + (v204_data * v280_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run1 = tensorforge::slmLoad<double, 32>(glb_m1 + (48_i32));
              tensorforge::intel_esimd::simd<double, 16> v285_data(glb_m1_run1.template select<16, 1>(0));
              double v286_data = s0_w0[3];
              tensorforge::intel_esimd::simd<double, 16> v288_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v288_data + (v285_data * v286_data));
              double v291_data = s0_w0[19];
              tensorforge::intel_esimd::simd<double, 16> v293_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v293_data + (v285_data * v291_data));
              double v296_data = s0_w0[35];
              tensorforge::intel_esimd::simd<double, 16> v298_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v298_data + (v285_data * v296_data));
              double v301_data = s0_w0[51];
              tensorforge::intel_esimd::simd<double, 16> v303_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v303_data + (v285_data * v301_data));
              double v306_data = s0_w0[67];
              tensorforge::intel_esimd::simd<double, 16> v308_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v308_data + (v285_data * v306_data));
              double v311_data = s0_w0[83];
              tensorforge::intel_esimd::simd<double, 16> v313_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v313_data + (v285_data * v311_data));
              double v316_data = s0_w0[99];
              tensorforge::intel_esimd::simd<double, 16> v318_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v318_data + (v285_data * v316_data));
              double v321_data = s0_w0[115];
              tensorforge::intel_esimd::simd<double, 16> v323_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v323_data + (v285_data * v321_data));
              double v326_data = s0_w0[131];
              tensorforge::intel_esimd::simd<double, 16> v328_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v328_data + (v285_data * v326_data));
              double v331_data = s0_w0[147];
              tensorforge::intel_esimd::simd<double, 16> v333_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v333_data + (v285_data * v331_data));
              double v336_data = s0_w0[163];
              tensorforge::intel_esimd::simd<double, 16> v338_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v338_data + (v285_data * v336_data));
              double v341_data = s0_w0[179];
              tensorforge::intel_esimd::simd<double, 16> v343_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v343_data + (v285_data * v341_data));
              double v346_data = s0_w0[195];
              tensorforge::intel_esimd::simd<double, 16> v348_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v348_data + (v285_data * v346_data));
              double v351_data = s0_w0[211];
              tensorforge::intel_esimd::simd<double, 16> v353_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v353_data + (v285_data * v351_data));
              double v356_data = s0_w0[227];
              tensorforge::intel_esimd::simd<double, 16> v358_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v358_data + (v285_data * v356_data));
              double v361_data = s0_w0[243];
              tensorforge::intel_esimd::simd<double, 16> v363_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v363_data + (v285_data * v361_data));
              tensorforge::intel_esimd::simd<double, 16> v366_data(glb_m1_run1.template select<16, 1>(16));
              double v367_data = s0_w0[4];
              tensorforge::intel_esimd::simd<double, 16> v369_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v369_data + (v366_data * v367_data));
              double v372_data = s0_w0[20];
              tensorforge::intel_esimd::simd<double, 16> v374_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v374_data + (v366_data * v372_data));
              double v377_data = s0_w0[36];
              tensorforge::intel_esimd::simd<double, 16> v379_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v379_data + (v366_data * v377_data));
              double v382_data = s0_w0[52];
              tensorforge::intel_esimd::simd<double, 16> v384_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v384_data + (v366_data * v382_data));
              double v387_data = s0_w0[68];
              tensorforge::intel_esimd::simd<double, 16> v389_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v389_data + (v366_data * v387_data));
              double v392_data = s0_w0[84];
              tensorforge::intel_esimd::simd<double, 16> v394_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v394_data + (v366_data * v392_data));
              double v397_data = s0_w0[100];
              tensorforge::intel_esimd::simd<double, 16> v399_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v399_data + (v366_data * v397_data));
              double v402_data = s0_w0[116];
              tensorforge::intel_esimd::simd<double, 16> v404_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v404_data + (v366_data * v402_data));
              double v407_data = s0_w0[132];
              tensorforge::intel_esimd::simd<double, 16> v409_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v409_data + (v366_data * v407_data));
              double v412_data = s0_w0[148];
              tensorforge::intel_esimd::simd<double, 16> v414_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v414_data + (v366_data * v412_data));
              double v417_data = s0_w0[164];
              tensorforge::intel_esimd::simd<double, 16> v419_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v419_data + (v366_data * v417_data));
              double v422_data = s0_w0[180];
              tensorforge::intel_esimd::simd<double, 16> v424_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v424_data + (v366_data * v422_data));
              double v427_data = s0_w0[196];
              tensorforge::intel_esimd::simd<double, 16> v429_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v429_data + (v366_data * v427_data));
              double v432_data = s0_w0[212];
              tensorforge::intel_esimd::simd<double, 16> v434_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v434_data + (v366_data * v432_data));
              double v437_data = s0_w0[228];
              tensorforge::intel_esimd::simd<double, 16> v439_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v439_data + (v366_data * v437_data));
              double v442_data = s0_w0[244];
              tensorforge::intel_esimd::simd<double, 16> v444_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v444_data + (v366_data * v442_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run2 = tensorforge::slmLoad<double, 32>(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<double, 16> v447_data(glb_m1_run2.template select<16, 1>(0));
              double v448_data = s0_w0[5];
              tensorforge::intel_esimd::simd<double, 16> v450_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v450_data + (v447_data * v448_data));
              double v453_data = s0_w0[21];
              tensorforge::intel_esimd::simd<double, 16> v455_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v455_data + (v447_data * v453_data));
              double v458_data = s0_w0[37];
              tensorforge::intel_esimd::simd<double, 16> v460_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v460_data + (v447_data * v458_data));
              double v463_data = s0_w0[53];
              tensorforge::intel_esimd::simd<double, 16> v465_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v465_data + (v447_data * v463_data));
              double v468_data = s0_w0[69];
              tensorforge::intel_esimd::simd<double, 16> v470_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v470_data + (v447_data * v468_data));
              double v473_data = s0_w0[85];
              tensorforge::intel_esimd::simd<double, 16> v475_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v475_data + (v447_data * v473_data));
              double v478_data = s0_w0[101];
              tensorforge::intel_esimd::simd<double, 16> v480_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v480_data + (v447_data * v478_data));
              double v483_data = s0_w0[117];
              tensorforge::intel_esimd::simd<double, 16> v485_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v485_data + (v447_data * v483_data));
              double v488_data = s0_w0[133];
              tensorforge::intel_esimd::simd<double, 16> v490_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v490_data + (v447_data * v488_data));
              double v493_data = s0_w0[149];
              tensorforge::intel_esimd::simd<double, 16> v495_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v495_data + (v447_data * v493_data));
              double v498_data = s0_w0[165];
              tensorforge::intel_esimd::simd<double, 16> v500_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v500_data + (v447_data * v498_data));
              double v503_data = s0_w0[181];
              tensorforge::intel_esimd::simd<double, 16> v505_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v505_data + (v447_data * v503_data));
              double v508_data = s0_w0[197];
              tensorforge::intel_esimd::simd<double, 16> v510_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v510_data + (v447_data * v508_data));
              double v513_data = s0_w0[213];
              tensorforge::intel_esimd::simd<double, 16> v515_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v515_data + (v447_data * v513_data));
              double v518_data = s0_w0[229];
              tensorforge::intel_esimd::simd<double, 16> v520_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v520_data + (v447_data * v518_data));
              double v523_data = s0_w0[245];
              tensorforge::intel_esimd::simd<double, 16> v525_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v525_data + (v447_data * v523_data));
              tensorforge::intel_esimd::simd<double, 16> v528_data(glb_m1_run2.template select<16, 1>(16));
              double v529_data = s0_w0[6];
              tensorforge::intel_esimd::simd<double, 16> v531_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v531_data + (v528_data * v529_data));
              double v534_data = s0_w0[22];
              tensorforge::intel_esimd::simd<double, 16> v536_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v536_data + (v528_data * v534_data));
              double v539_data = s0_w0[38];
              tensorforge::intel_esimd::simd<double, 16> v541_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v541_data + (v528_data * v539_data));
              double v544_data = s0_w0[54];
              tensorforge::intel_esimd::simd<double, 16> v546_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v546_data + (v528_data * v544_data));
              double v549_data = s0_w0[70];
              tensorforge::intel_esimd::simd<double, 16> v551_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v551_data + (v528_data * v549_data));
              double v554_data = s0_w0[86];
              tensorforge::intel_esimd::simd<double, 16> v556_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v556_data + (v528_data * v554_data));
              double v559_data = s0_w0[102];
              tensorforge::intel_esimd::simd<double, 16> v561_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v561_data + (v528_data * v559_data));
              double v564_data = s0_w0[118];
              tensorforge::intel_esimd::simd<double, 16> v566_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v566_data + (v528_data * v564_data));
              double v569_data = s0_w0[134];
              tensorforge::intel_esimd::simd<double, 16> v571_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v571_data + (v528_data * v569_data));
              double v574_data = s0_w0[150];
              tensorforge::intel_esimd::simd<double, 16> v576_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v576_data + (v528_data * v574_data));
              double v579_data = s0_w0[166];
              tensorforge::intel_esimd::simd<double, 16> v581_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v581_data + (v528_data * v579_data));
              double v584_data = s0_w0[182];
              tensorforge::intel_esimd::simd<double, 16> v586_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v586_data + (v528_data * v584_data));
              double v589_data = s0_w0[198];
              tensorforge::intel_esimd::simd<double, 16> v591_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v591_data + (v528_data * v589_data));
              double v594_data = s0_w0[214];
              tensorforge::intel_esimd::simd<double, 16> v596_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v596_data + (v528_data * v594_data));
              double v599_data = s0_w0[230];
              tensorforge::intel_esimd::simd<double, 16> v601_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v601_data + (v528_data * v599_data));
              double v604_data = s0_w0[246];
              tensorforge::intel_esimd::simd<double, 16> v606_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v606_data + (v528_data * v604_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run3 = tensorforge::slmLoad<double, 32>(glb_m1 + (112_i32));
              tensorforge::intel_esimd::simd<double, 16> v609_data(glb_m1_run3.template select<16, 1>(0));
              double v610_data = s0_w0[7];
              tensorforge::intel_esimd::simd<double, 16> v612_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v612_data + (v609_data * v610_data));
              double v615_data = s0_w0[23];
              tensorforge::intel_esimd::simd<double, 16> v617_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v617_data + (v609_data * v615_data));
              double v620_data = s0_w0[39];
              tensorforge::intel_esimd::simd<double, 16> v622_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v622_data + (v609_data * v620_data));
              double v625_data = s0_w0[55];
              tensorforge::intel_esimd::simd<double, 16> v627_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v627_data + (v609_data * v625_data));
              double v630_data = s0_w0[71];
              tensorforge::intel_esimd::simd<double, 16> v632_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v632_data + (v609_data * v630_data));
              double v635_data = s0_w0[87];
              tensorforge::intel_esimd::simd<double, 16> v637_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v637_data + (v609_data * v635_data));
              double v640_data = s0_w0[103];
              tensorforge::intel_esimd::simd<double, 16> v642_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v642_data + (v609_data * v640_data));
              double v645_data = s0_w0[119];
              tensorforge::intel_esimd::simd<double, 16> v647_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v647_data + (v609_data * v645_data));
              double v650_data = s0_w0[135];
              tensorforge::intel_esimd::simd<double, 16> v652_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v652_data + (v609_data * v650_data));
              double v655_data = s0_w0[151];
              tensorforge::intel_esimd::simd<double, 16> v657_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v657_data + (v609_data * v655_data));
              double v660_data = s0_w0[167];
              tensorforge::intel_esimd::simd<double, 16> v662_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v662_data + (v609_data * v660_data));
              double v665_data = s0_w0[183];
              tensorforge::intel_esimd::simd<double, 16> v667_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v667_data + (v609_data * v665_data));
              double v670_data = s0_w0[199];
              tensorforge::intel_esimd::simd<double, 16> v672_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v672_data + (v609_data * v670_data));
              double v675_data = s0_w0[215];
              tensorforge::intel_esimd::simd<double, 16> v677_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v677_data + (v609_data * v675_data));
              double v680_data = s0_w0[231];
              tensorforge::intel_esimd::simd<double, 16> v682_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v682_data + (v609_data * v680_data));
              double v685_data = s0_w0[247];
              tensorforge::intel_esimd::simd<double, 16> v687_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v687_data + (v609_data * v685_data));
              tensorforge::intel_esimd::simd<double, 16> v690_data(glb_m1_run3.template select<16, 1>(16));
              double v691_data = s0_w0[8];
              tensorforge::intel_esimd::simd<double, 16> v693_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v693_data + (v690_data * v691_data));
              double v696_data = s0_w0[24];
              tensorforge::intel_esimd::simd<double, 16> v698_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v698_data + (v690_data * v696_data));
              double v701_data = s0_w0[40];
              tensorforge::intel_esimd::simd<double, 16> v703_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v703_data + (v690_data * v701_data));
              double v706_data = s0_w0[56];
              tensorforge::intel_esimd::simd<double, 16> v708_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v708_data + (v690_data * v706_data));
              double v711_data = s0_w0[72];
              tensorforge::intel_esimd::simd<double, 16> v713_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v713_data + (v690_data * v711_data));
              double v716_data = s0_w0[88];
              tensorforge::intel_esimd::simd<double, 16> v718_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v718_data + (v690_data * v716_data));
              double v721_data = s0_w0[104];
              tensorforge::intel_esimd::simd<double, 16> v723_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v723_data + (v690_data * v721_data));
              double v726_data = s0_w0[120];
              tensorforge::intel_esimd::simd<double, 16> v728_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v728_data + (v690_data * v726_data));
              double v731_data = s0_w0[136];
              tensorforge::intel_esimd::simd<double, 16> v733_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v733_data + (v690_data * v731_data));
              double v736_data = s0_w0[152];
              tensorforge::intel_esimd::simd<double, 16> v738_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v738_data + (v690_data * v736_data));
              double v741_data = s0_w0[168];
              tensorforge::intel_esimd::simd<double, 16> v743_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v743_data + (v690_data * v741_data));
              double v746_data = s0_w0[184];
              tensorforge::intel_esimd::simd<double, 16> v748_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v748_data + (v690_data * v746_data));
              double v751_data = s0_w0[200];
              tensorforge::intel_esimd::simd<double, 16> v753_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v753_data + (v690_data * v751_data));
              double v756_data = s0_w0[216];
              tensorforge::intel_esimd::simd<double, 16> v758_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v758_data + (v690_data * v756_data));
              double v761_data = s0_w0[232];
              tensorforge::intel_esimd::simd<double, 16> v763_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v763_data + (v690_data * v761_data));
              double v766_data = s0_w0[248];
              tensorforge::intel_esimd::simd<double, 16> v768_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v768_data + (v690_data * v766_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run4 = tensorforge::slmLoad<double, 32>(glb_m1 + (144_i32));
              tensorforge::intel_esimd::simd<double, 16> v771_data(glb_m1_run4.template select<16, 1>(0));
              double v772_data = s0_w0[9];
              tensorforge::intel_esimd::simd<double, 16> v774_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v774_data + (v771_data * v772_data));
              double v777_data = s0_w0[25];
              tensorforge::intel_esimd::simd<double, 16> v779_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v779_data + (v771_data * v777_data));
              double v782_data = s0_w0[41];
              tensorforge::intel_esimd::simd<double, 16> v784_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v784_data + (v771_data * v782_data));
              double v787_data = s0_w0[57];
              tensorforge::intel_esimd::simd<double, 16> v789_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v789_data + (v771_data * v787_data));
              double v792_data = s0_w0[73];
              tensorforge::intel_esimd::simd<double, 16> v794_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v794_data + (v771_data * v792_data));
              double v797_data = s0_w0[89];
              tensorforge::intel_esimd::simd<double, 16> v799_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v799_data + (v771_data * v797_data));
              double v802_data = s0_w0[105];
              tensorforge::intel_esimd::simd<double, 16> v804_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v804_data + (v771_data * v802_data));
              double v807_data = s0_w0[121];
              tensorforge::intel_esimd::simd<double, 16> v809_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v809_data + (v771_data * v807_data));
              double v812_data = s0_w0[137];
              tensorforge::intel_esimd::simd<double, 16> v814_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v814_data + (v771_data * v812_data));
              double v817_data = s0_w0[153];
              tensorforge::intel_esimd::simd<double, 16> v819_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v819_data + (v771_data * v817_data));
              double v822_data = s0_w0[169];
              tensorforge::intel_esimd::simd<double, 16> v824_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v824_data + (v771_data * v822_data));
              double v827_data = s0_w0[185];
              tensorforge::intel_esimd::simd<double, 16> v829_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v829_data + (v771_data * v827_data));
              double v832_data = s0_w0[201];
              tensorforge::intel_esimd::simd<double, 16> v834_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v834_data + (v771_data * v832_data));
              double v837_data = s0_w0[217];
              tensorforge::intel_esimd::simd<double, 16> v839_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v839_data + (v771_data * v837_data));
              double v842_data = s0_w0[233];
              tensorforge::intel_esimd::simd<double, 16> v844_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v844_data + (v771_data * v842_data));
              double v847_data = s0_w0[249];
              tensorforge::intel_esimd::simd<double, 16> v849_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v849_data + (v771_data * v847_data));
              tensorforge::intel_esimd::simd<double, 16> v852_data(glb_m1_run4.template select<16, 1>(16));
              double v853_data = s0_w0[10];
              tensorforge::intel_esimd::simd<double, 16> v855_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v855_data + (v852_data * v853_data));
              double v858_data = s0_w0[26];
              tensorforge::intel_esimd::simd<double, 16> v860_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v860_data + (v852_data * v858_data));
              double v863_data = s0_w0[42];
              tensorforge::intel_esimd::simd<double, 16> v865_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v865_data + (v852_data * v863_data));
              double v868_data = s0_w0[58];
              tensorforge::intel_esimd::simd<double, 16> v870_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v870_data + (v852_data * v868_data));
              double v873_data = s0_w0[74];
              tensorforge::intel_esimd::simd<double, 16> v875_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v875_data + (v852_data * v873_data));
              double v878_data = s0_w0[90];
              tensorforge::intel_esimd::simd<double, 16> v880_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v880_data + (v852_data * v878_data));
              double v883_data = s0_w0[106];
              tensorforge::intel_esimd::simd<double, 16> v885_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v885_data + (v852_data * v883_data));
              double v888_data = s0_w0[122];
              tensorforge::intel_esimd::simd<double, 16> v890_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v890_data + (v852_data * v888_data));
              double v893_data = s0_w0[138];
              tensorforge::intel_esimd::simd<double, 16> v895_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v895_data + (v852_data * v893_data));
              double v898_data = s0_w0[154];
              tensorforge::intel_esimd::simd<double, 16> v900_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v900_data + (v852_data * v898_data));
              double v903_data = s0_w0[170];
              tensorforge::intel_esimd::simd<double, 16> v905_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v905_data + (v852_data * v903_data));
              double v908_data = s0_w0[186];
              tensorforge::intel_esimd::simd<double, 16> v910_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v910_data + (v852_data * v908_data));
              double v913_data = s0_w0[202];
              tensorforge::intel_esimd::simd<double, 16> v915_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v915_data + (v852_data * v913_data));
              double v918_data = s0_w0[218];
              tensorforge::intel_esimd::simd<double, 16> v920_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v920_data + (v852_data * v918_data));
              double v923_data = s0_w0[234];
              tensorforge::intel_esimd::simd<double, 16> v925_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v925_data + (v852_data * v923_data));
              double v928_data = s0_w0[250];
              tensorforge::intel_esimd::simd<double, 16> v930_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v930_data + (v852_data * v928_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run5 = tensorforge::slmLoad<double, 32>(glb_m1 + (176_i32));
              tensorforge::intel_esimd::simd<double, 16> v933_data(glb_m1_run5.template select<16, 1>(0));
              double v934_data = s0_w0[11];
              tensorforge::intel_esimd::simd<double, 16> v936_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v936_data + (v933_data * v934_data));
              double v939_data = s0_w0[27];
              tensorforge::intel_esimd::simd<double, 16> v941_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v941_data + (v933_data * v939_data));
              double v944_data = s0_w0[43];
              tensorforge::intel_esimd::simd<double, 16> v946_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v946_data + (v933_data * v944_data));
              double v949_data = s0_w0[59];
              tensorforge::intel_esimd::simd<double, 16> v951_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v951_data + (v933_data * v949_data));
              double v954_data = s0_w0[75];
              tensorforge::intel_esimd::simd<double, 16> v956_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v956_data + (v933_data * v954_data));
              double v959_data = s0_w0[91];
              tensorforge::intel_esimd::simd<double, 16> v961_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v961_data + (v933_data * v959_data));
              double v964_data = s0_w0[107];
              tensorforge::intel_esimd::simd<double, 16> v966_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v966_data + (v933_data * v964_data));
              double v969_data = s0_w0[123];
              tensorforge::intel_esimd::simd<double, 16> v971_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v971_data + (v933_data * v969_data));
              double v974_data = s0_w0[139];
              tensorforge::intel_esimd::simd<double, 16> v976_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v976_data + (v933_data * v974_data));
              double v979_data = s0_w0[155];
              tensorforge::intel_esimd::simd<double, 16> v981_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v981_data + (v933_data * v979_data));
              double v984_data = s0_w0[171];
              tensorforge::intel_esimd::simd<double, 16> v986_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v986_data + (v933_data * v984_data));
              double v989_data = s0_w0[187];
              tensorforge::intel_esimd::simd<double, 16> v991_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v991_data + (v933_data * v989_data));
              double v994_data = s0_w0[203];
              tensorforge::intel_esimd::simd<double, 16> v996_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v996_data + (v933_data * v994_data));
              double v999_data = s0_w0[219];
              tensorforge::intel_esimd::simd<double, 16> v1001_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1001_data + (v933_data * v999_data));
              double v1004_data = s0_w0[235];
              tensorforge::intel_esimd::simd<double, 16> v1006_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1006_data + (v933_data * v1004_data));
              double v1009_data = s0_w0[251];
              tensorforge::intel_esimd::simd<double, 16> v1011_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1011_data + (v933_data * v1009_data));
              tensorforge::intel_esimd::simd<double, 16> v1014_data(glb_m1_run5.template select<16, 1>(16));
              double v1015_data = s0_w0[12];
              tensorforge::intel_esimd::simd<double, 16> v1017_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v1017_data + (v1014_data * v1015_data));
              double v1020_data = s0_w0[28];
              tensorforge::intel_esimd::simd<double, 16> v1022_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v1022_data + (v1014_data * v1020_data));
              double v1025_data = s0_w0[44];
              tensorforge::intel_esimd::simd<double, 16> v1027_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v1027_data + (v1014_data * v1025_data));
              double v1030_data = s0_w0[60];
              tensorforge::intel_esimd::simd<double, 16> v1032_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v1032_data + (v1014_data * v1030_data));
              double v1035_data = s0_w0[76];
              tensorforge::intel_esimd::simd<double, 16> v1037_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v1037_data + (v1014_data * v1035_data));
              double v1040_data = s0_w0[92];
              tensorforge::intel_esimd::simd<double, 16> v1042_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v1042_data + (v1014_data * v1040_data));
              double v1045_data = s0_w0[108];
              tensorforge::intel_esimd::simd<double, 16> v1047_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v1047_data + (v1014_data * v1045_data));
              double v1050_data = s0_w0[124];
              tensorforge::intel_esimd::simd<double, 16> v1052_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v1052_data + (v1014_data * v1050_data));
              double v1055_data = s0_w0[140];
              tensorforge::intel_esimd::simd<double, 16> v1057_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v1057_data + (v1014_data * v1055_data));
              double v1060_data = s0_w0[156];
              tensorforge::intel_esimd::simd<double, 16> v1062_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v1062_data + (v1014_data * v1060_data));
              double v1065_data = s0_w0[172];
              tensorforge::intel_esimd::simd<double, 16> v1067_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v1067_data + (v1014_data * v1065_data));
              double v1070_data = s0_w0[188];
              tensorforge::intel_esimd::simd<double, 16> v1072_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v1072_data + (v1014_data * v1070_data));
              double v1075_data = s0_w0[204];
              tensorforge::intel_esimd::simd<double, 16> v1077_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v1077_data + (v1014_data * v1075_data));
              double v1080_data = s0_w0[220];
              tensorforge::intel_esimd::simd<double, 16> v1082_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1082_data + (v1014_data * v1080_data));
              double v1085_data = s0_w0[236];
              tensorforge::intel_esimd::simd<double, 16> v1087_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1087_data + (v1014_data * v1085_data));
              double v1090_data = s0_w0[252];
              tensorforge::intel_esimd::simd<double, 16> v1092_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1092_data + (v1014_data * v1090_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run6 = tensorforge::slmLoad<double, 32>(glb_m1 + (208_i32));
              tensorforge::intel_esimd::simd<double, 16> v1095_data(glb_m1_run6.template select<16, 1>(0));
              double v1096_data = s0_w0[13];
              tensorforge::intel_esimd::simd<double, 16> v1098_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v1098_data + (v1095_data * v1096_data));
              double v1101_data = s0_w0[29];
              tensorforge::intel_esimd::simd<double, 16> v1103_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v1103_data + (v1095_data * v1101_data));
              double v1106_data = s0_w0[45];
              tensorforge::intel_esimd::simd<double, 16> v1108_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v1108_data + (v1095_data * v1106_data));
              double v1111_data = s0_w0[61];
              tensorforge::intel_esimd::simd<double, 16> v1113_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v1113_data + (v1095_data * v1111_data));
              double v1116_data = s0_w0[77];
              tensorforge::intel_esimd::simd<double, 16> v1118_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v1118_data + (v1095_data * v1116_data));
              double v1121_data = s0_w0[93];
              tensorforge::intel_esimd::simd<double, 16> v1123_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v1123_data + (v1095_data * v1121_data));
              double v1126_data = s0_w0[109];
              tensorforge::intel_esimd::simd<double, 16> v1128_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v1128_data + (v1095_data * v1126_data));
              double v1131_data = s0_w0[125];
              tensorforge::intel_esimd::simd<double, 16> v1133_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v1133_data + (v1095_data * v1131_data));
              double v1136_data = s0_w0[141];
              tensorforge::intel_esimd::simd<double, 16> v1138_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v1138_data + (v1095_data * v1136_data));
              double v1141_data = s0_w0[157];
              tensorforge::intel_esimd::simd<double, 16> v1143_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v1143_data + (v1095_data * v1141_data));
              double v1146_data = s0_w0[173];
              tensorforge::intel_esimd::simd<double, 16> v1148_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v1148_data + (v1095_data * v1146_data));
              double v1151_data = s0_w0[189];
              tensorforge::intel_esimd::simd<double, 16> v1153_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v1153_data + (v1095_data * v1151_data));
              double v1156_data = s0_w0[205];
              tensorforge::intel_esimd::simd<double, 16> v1158_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v1158_data + (v1095_data * v1156_data));
              double v1161_data = s0_w0[221];
              tensorforge::intel_esimd::simd<double, 16> v1163_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1163_data + (v1095_data * v1161_data));
              double v1166_data = s0_w0[237];
              tensorforge::intel_esimd::simd<double, 16> v1168_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1168_data + (v1095_data * v1166_data));
              double v1171_data = s0_w0[253];
              tensorforge::intel_esimd::simd<double, 16> v1173_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1173_data + (v1095_data * v1171_data));
              tensorforge::intel_esimd::simd<double, 16> v1176_data(glb_m1_run6.template select<16, 1>(16));
              double v1177_data = s0_w0[14];
              tensorforge::intel_esimd::simd<double, 16> v1179_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v1179_data + (v1176_data * v1177_data));
              double v1182_data = s0_w0[30];
              tensorforge::intel_esimd::simd<double, 16> v1184_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v1184_data + (v1176_data * v1182_data));
              double v1187_data = s0_w0[46];
              tensorforge::intel_esimd::simd<double, 16> v1189_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v1189_data + (v1176_data * v1187_data));
              double v1192_data = s0_w0[62];
              tensorforge::intel_esimd::simd<double, 16> v1194_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v1194_data + (v1176_data * v1192_data));
              double v1197_data = s0_w0[78];
              tensorforge::intel_esimd::simd<double, 16> v1199_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v1199_data + (v1176_data * v1197_data));
              double v1202_data = s0_w0[94];
              tensorforge::intel_esimd::simd<double, 16> v1204_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v1204_data + (v1176_data * v1202_data));
              double v1207_data = s0_w0[110];
              tensorforge::intel_esimd::simd<double, 16> v1209_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v1209_data + (v1176_data * v1207_data));
              double v1212_data = s0_w0[126];
              tensorforge::intel_esimd::simd<double, 16> v1214_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v1214_data + (v1176_data * v1212_data));
              double v1217_data = s0_w0[142];
              tensorforge::intel_esimd::simd<double, 16> v1219_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v1219_data + (v1176_data * v1217_data));
              double v1222_data = s0_w0[158];
              tensorforge::intel_esimd::simd<double, 16> v1224_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v1224_data + (v1176_data * v1222_data));
              double v1227_data = s0_w0[174];
              tensorforge::intel_esimd::simd<double, 16> v1229_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v1229_data + (v1176_data * v1227_data));
              double v1232_data = s0_w0[190];
              tensorforge::intel_esimd::simd<double, 16> v1234_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v1234_data + (v1176_data * v1232_data));
              double v1237_data = s0_w0[206];
              tensorforge::intel_esimd::simd<double, 16> v1239_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v1239_data + (v1176_data * v1237_data));
              double v1242_data = s0_w0[222];
              tensorforge::intel_esimd::simd<double, 16> v1244_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1244_data + (v1176_data * v1242_data));
              double v1247_data = s0_w0[238];
              tensorforge::intel_esimd::simd<double, 16> v1249_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1249_data + (v1176_data * v1247_data));
              double v1252_data = s0_w0[254];
              tensorforge::intel_esimd::simd<double, 16> v1254_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1254_data + (v1176_data * v1252_data));
              tensorforge::intel_esimd::simd<double, 16> v1257_data = tensorforge::slmLoad<double, 16>(glb_m1 + (240_i32));
              double v1258_data = s0_w0[15];
              tensorforge::intel_esimd::simd<double, 16> v1260_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v1260_data + (v1257_data * v1258_data));
              double v1263_data = s0_w0[31];
              tensorforge::intel_esimd::simd<double, 16> v1265_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v1265_data + (v1257_data * v1263_data));
              double v1268_data = s0_w0[47];
              tensorforge::intel_esimd::simd<double, 16> v1270_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v1270_data + (v1257_data * v1268_data));
              double v1273_data = s0_w0[63];
              tensorforge::intel_esimd::simd<double, 16> v1275_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v1275_data + (v1257_data * v1273_data));
              double v1278_data = s0_w0[79];
              tensorforge::intel_esimd::simd<double, 16> v1280_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v1280_data + (v1257_data * v1278_data));
              double v1283_data = s0_w0[95];
              tensorforge::intel_esimd::simd<double, 16> v1285_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v1285_data + (v1257_data * v1283_data));
              double v1288_data = s0_w0[111];
              tensorforge::intel_esimd::simd<double, 16> v1290_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v1290_data + (v1257_data * v1288_data));
              double v1293_data = s0_w0[127];
              tensorforge::intel_esimd::simd<double, 16> v1295_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v1295_data + (v1257_data * v1293_data));
              double v1298_data = s0_w0[143];
              tensorforge::intel_esimd::simd<double, 16> v1300_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v1300_data + (v1257_data * v1298_data));
              double v1303_data = s0_w0[159];
              tensorforge::intel_esimd::simd<double, 16> v1305_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v1305_data + (v1257_data * v1303_data));
              double v1308_data = s0_w0[175];
              tensorforge::intel_esimd::simd<double, 16> v1310_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v1310_data + (v1257_data * v1308_data));
              double v1313_data = s0_w0[191];
              tensorforge::intel_esimd::simd<double, 16> v1315_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v1315_data + (v1257_data * v1313_data));
              double v1318_data = s0_w0[207];
              tensorforge::intel_esimd::simd<double, 16> v1320_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v1320_data + (v1257_data * v1318_data));
              double v1323_data = s0_w0[223];
              tensorforge::intel_esimd::simd<double, 16> v1325_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1325_data + (v1257_data * v1323_data));
              double v1328_data = s0_w0[239];
              tensorforge::intel_esimd::simd<double, 16> v1330_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1330_data + (v1257_data * v1328_data));
              double v1333_data = s0_w0[255];
              tensorforge::intel_esimd::simd<double, 16> v1335_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1335_data + (v1257_data * v1333_data));
              // r0 = ir0
              #pragma unroll
              for (int32_t v1337_n0 = 0; v1337_n0 < 1; ++v1337_n0) {
                int32_t v1339_a = v1337_n0 * 16;
                #pragma unroll
                for (int32_t v1338_n1 = 0; v1338_n1 < 16; ++v1338_n1) {
                  int32_t v1341_a = v1339_a + (v1338_n1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v1342_data(ir0.template select<16, 1>(v1341_a));
                  r0.template select<16, 1>(v1341_a) = v1342_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v1343_i0 = 0; v1343_i0 < 1; ++v1343_i0) {
                int32_t v1345_a = v1343_i0 * 16;
                #pragma unroll
                for (int32_t v1344_i1 = 0; v1344_i1 < 16; ++v1344_i1) {
                  int32_t v1347_a = v1345_a + (v1344_i1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v1348_data(r0.template select<16, 1>(v1347_a));
                  v1348_data.copy_to(glb_m0 + (v1347_a));
                }
              }
            }
          }
        }
      }
    });
  });
}

