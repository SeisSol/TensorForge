// === base name ===
kernel_d7061a999c4623a6

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d7061a999c4623a6 = {{1, 32, 1}, 16, 16, 1, 32, 71680, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d7061a999c4623a6(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d7061a999c4623a6(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d7061a999c4623a6(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_d7061a999c4623a6(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d7061a999c4623a6(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d7061a999c4623a6(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d7061a999c4623a6(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<double> tempShrMem = localShrMem0 + (256);
          const double *const __restrict__ ptr_glb_m1 = &m1[0];
          tensorforge::SlmPtr<double> glb_m1 = totalShrMem + (0);
          // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<double, 16> v12_ld;
            v12_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v12_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<double, 16> v13_ld;
            v13_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v13_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<double, 16> v14_ld;
            v14_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v14_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<double, 16> v15_ld;
            v15_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v15_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<double, 16> v16_ld;
            v16_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v16_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<double, 16> v17_ld;
            v17_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v17_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<double, 16> v18_ld;
            v18_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v18_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<double, 16> v19_ld;
            v19_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v19_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<double, 16> v20_ld;
            v20_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v20_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<double, 16> v21_ld;
            v21_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v21_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<double, 16> v22_ld;
            v22_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v22_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<double, 16> v23_ld;
            v23_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v23_ld);
          }
          if (item.get_local_id(1) == 12) {
            tensorforge::intel_esimd::simd<double, 16> v24_ld;
            v24_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v24_ld);
          }
          if (item.get_local_id(1) == 13) {
            tensorforge::intel_esimd::simd<double, 16> v25_ld;
            v25_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v25_ld);
          }
          if (item.get_local_id(1) == 14) {
            tensorforge::intel_esimd::simd<double, 16> v26_ld;
            v26_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v26_ld);
          }
          if (item.get_local_id(1) == 15) {
            tensorforge::intel_esimd::simd<double, 16> v27_ld;
            v27_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v27_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<double> s0 = localShrMem0 + (0);
          for (size_t v29_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v29_batchId0 < numElements0; v29_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v30_ahead1 = v29_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v32_batchId1 = (v30_ahead1 < numElements0) ? v30_ahead1 : v29_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v29_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v29_batchId0 * 256 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v29_batchId0 * 256 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              #pragma unroll
              for (int32_t i = 0; i < 16; i += 2) {
                tensorforge::intel_esimd::simd<double, 32> v39_ld;
                v39_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + i * 16));
                tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + i * 16), v39_ld);
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<double, 256> r0(0.0);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<double, 256> ir0(0.0);
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run0 = tensorforge::slmLoad<double, 32>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<double, 16> v45_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 256> s0_w0 = tensorforge::slmLoad<double, 256>(s0 + 0);
              double v46_data = s0_w0[0];
              tensorforge::intel_esimd::simd<double, 16> v48_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v48_data + (v45_data * v46_data));
              double v51_data = s0_w0[16];
              tensorforge::intel_esimd::simd<double, 16> v53_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v53_data + (v45_data * v51_data));
              double v56_data = s0_w0[32];
              tensorforge::intel_esimd::simd<double, 16> v58_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v58_data + (v45_data * v56_data));
              double v61_data = s0_w0[48];
              tensorforge::intel_esimd::simd<double, 16> v63_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v63_data + (v45_data * v61_data));
              double v66_data = s0_w0[64];
              tensorforge::intel_esimd::simd<double, 16> v68_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v68_data + (v45_data * v66_data));
              double v71_data = s0_w0[80];
              tensorforge::intel_esimd::simd<double, 16> v73_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v73_data + (v45_data * v71_data));
              double v76_data = s0_w0[96];
              tensorforge::intel_esimd::simd<double, 16> v78_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v78_data + (v45_data * v76_data));
              double v81_data = s0_w0[112];
              tensorforge::intel_esimd::simd<double, 16> v83_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v83_data + (v45_data * v81_data));
              double v86_data = s0_w0[128];
              tensorforge::intel_esimd::simd<double, 16> v88_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v88_data + (v45_data * v86_data));
              double v91_data = s0_w0[144];
              tensorforge::intel_esimd::simd<double, 16> v93_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v93_data + (v45_data * v91_data));
              double v96_data = s0_w0[160];
              tensorforge::intel_esimd::simd<double, 16> v98_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v98_data + (v45_data * v96_data));
              double v101_data = s0_w0[176];
              tensorforge::intel_esimd::simd<double, 16> v103_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v103_data + (v45_data * v101_data));
              double v106_data = s0_w0[192];
              tensorforge::intel_esimd::simd<double, 16> v108_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v108_data + (v45_data * v106_data));
              double v111_data = s0_w0[208];
              tensorforge::intel_esimd::simd<double, 16> v113_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v113_data + (v45_data * v111_data));
              double v116_data = s0_w0[224];
              tensorforge::intel_esimd::simd<double, 16> v118_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v118_data + (v45_data * v116_data));
              double v121_data = s0_w0[240];
              tensorforge::intel_esimd::simd<double, 16> v123_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v123_data + (v45_data * v121_data));
              tensorforge::intel_esimd::simd<double, 16> v126_data(glb_m1_run0.template select<16, 1>(16));
              double v127_data = s0_w0[1];
              tensorforge::intel_esimd::simd<double, 16> v129_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v129_data + (v126_data * v127_data));
              double v132_data = s0_w0[17];
              tensorforge::intel_esimd::simd<double, 16> v134_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v134_data + (v126_data * v132_data));
              double v137_data = s0_w0[33];
              tensorforge::intel_esimd::simd<double, 16> v139_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v139_data + (v126_data * v137_data));
              double v142_data = s0_w0[49];
              tensorforge::intel_esimd::simd<double, 16> v144_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v144_data + (v126_data * v142_data));
              double v147_data = s0_w0[65];
              tensorforge::intel_esimd::simd<double, 16> v149_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v149_data + (v126_data * v147_data));
              double v152_data = s0_w0[81];
              tensorforge::intel_esimd::simd<double, 16> v154_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v154_data + (v126_data * v152_data));
              double v157_data = s0_w0[97];
              tensorforge::intel_esimd::simd<double, 16> v159_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v159_data + (v126_data * v157_data));
              double v162_data = s0_w0[113];
              tensorforge::intel_esimd::simd<double, 16> v164_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v164_data + (v126_data * v162_data));
              double v167_data = s0_w0[129];
              tensorforge::intel_esimd::simd<double, 16> v169_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v169_data + (v126_data * v167_data));
              double v172_data = s0_w0[145];
              tensorforge::intel_esimd::simd<double, 16> v174_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v174_data + (v126_data * v172_data));
              double v177_data = s0_w0[161];
              tensorforge::intel_esimd::simd<double, 16> v179_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v179_data + (v126_data * v177_data));
              double v182_data = s0_w0[177];
              tensorforge::intel_esimd::simd<double, 16> v184_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v184_data + (v126_data * v182_data));
              double v187_data = s0_w0[193];
              tensorforge::intel_esimd::simd<double, 16> v189_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v189_data + (v126_data * v187_data));
              double v192_data = s0_w0[209];
              tensorforge::intel_esimd::simd<double, 16> v194_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v194_data + (v126_data * v192_data));
              double v197_data = s0_w0[225];
              tensorforge::intel_esimd::simd<double, 16> v199_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v199_data + (v126_data * v197_data));
              double v202_data = s0_w0[241];
              tensorforge::intel_esimd::simd<double, 16> v204_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v204_data + (v126_data * v202_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run1 = tensorforge::slmLoad<double, 32>(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<double, 16> v207_data(glb_m1_run1.template select<16, 1>(0));
              double v208_data = s0_w0[2];
              tensorforge::intel_esimd::simd<double, 16> v210_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v210_data + (v207_data * v208_data));
              double v213_data = s0_w0[18];
              tensorforge::intel_esimd::simd<double, 16> v215_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v215_data + (v207_data * v213_data));
              double v218_data = s0_w0[34];
              tensorforge::intel_esimd::simd<double, 16> v220_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v220_data + (v207_data * v218_data));
              double v223_data = s0_w0[50];
              tensorforge::intel_esimd::simd<double, 16> v225_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v225_data + (v207_data * v223_data));
              double v228_data = s0_w0[66];
              tensorforge::intel_esimd::simd<double, 16> v230_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v230_data + (v207_data * v228_data));
              double v233_data = s0_w0[82];
              tensorforge::intel_esimd::simd<double, 16> v235_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v235_data + (v207_data * v233_data));
              double v238_data = s0_w0[98];
              tensorforge::intel_esimd::simd<double, 16> v240_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v240_data + (v207_data * v238_data));
              double v243_data = s0_w0[114];
              tensorforge::intel_esimd::simd<double, 16> v245_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v245_data + (v207_data * v243_data));
              double v248_data = s0_w0[130];
              tensorforge::intel_esimd::simd<double, 16> v250_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v250_data + (v207_data * v248_data));
              double v253_data = s0_w0[146];
              tensorforge::intel_esimd::simd<double, 16> v255_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v255_data + (v207_data * v253_data));
              double v258_data = s0_w0[162];
              tensorforge::intel_esimd::simd<double, 16> v260_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v260_data + (v207_data * v258_data));
              double v263_data = s0_w0[178];
              tensorforge::intel_esimd::simd<double, 16> v265_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v265_data + (v207_data * v263_data));
              double v268_data = s0_w0[194];
              tensorforge::intel_esimd::simd<double, 16> v270_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v270_data + (v207_data * v268_data));
              double v273_data = s0_w0[210];
              tensorforge::intel_esimd::simd<double, 16> v275_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v275_data + (v207_data * v273_data));
              double v278_data = s0_w0[226];
              tensorforge::intel_esimd::simd<double, 16> v280_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v280_data + (v207_data * v278_data));
              double v283_data = s0_w0[242];
              tensorforge::intel_esimd::simd<double, 16> v285_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v285_data + (v207_data * v283_data));
              tensorforge::intel_esimd::simd<double, 16> v288_data(glb_m1_run1.template select<16, 1>(16));
              double v289_data = s0_w0[3];
              tensorforge::intel_esimd::simd<double, 16> v291_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v291_data + (v288_data * v289_data));
              double v294_data = s0_w0[19];
              tensorforge::intel_esimd::simd<double, 16> v296_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v296_data + (v288_data * v294_data));
              double v299_data = s0_w0[35];
              tensorforge::intel_esimd::simd<double, 16> v301_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v301_data + (v288_data * v299_data));
              double v304_data = s0_w0[51];
              tensorforge::intel_esimd::simd<double, 16> v306_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v306_data + (v288_data * v304_data));
              double v309_data = s0_w0[67];
              tensorforge::intel_esimd::simd<double, 16> v311_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v311_data + (v288_data * v309_data));
              double v314_data = s0_w0[83];
              tensorforge::intel_esimd::simd<double, 16> v316_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v316_data + (v288_data * v314_data));
              double v319_data = s0_w0[99];
              tensorforge::intel_esimd::simd<double, 16> v321_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v321_data + (v288_data * v319_data));
              double v324_data = s0_w0[115];
              tensorforge::intel_esimd::simd<double, 16> v326_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v326_data + (v288_data * v324_data));
              double v329_data = s0_w0[131];
              tensorforge::intel_esimd::simd<double, 16> v331_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v331_data + (v288_data * v329_data));
              double v334_data = s0_w0[147];
              tensorforge::intel_esimd::simd<double, 16> v336_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v336_data + (v288_data * v334_data));
              double v339_data = s0_w0[163];
              tensorforge::intel_esimd::simd<double, 16> v341_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v341_data + (v288_data * v339_data));
              double v344_data = s0_w0[179];
              tensorforge::intel_esimd::simd<double, 16> v346_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v346_data + (v288_data * v344_data));
              double v349_data = s0_w0[195];
              tensorforge::intel_esimd::simd<double, 16> v351_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v351_data + (v288_data * v349_data));
              double v354_data = s0_w0[211];
              tensorforge::intel_esimd::simd<double, 16> v356_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v356_data + (v288_data * v354_data));
              double v359_data = s0_w0[227];
              tensorforge::intel_esimd::simd<double, 16> v361_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v361_data + (v288_data * v359_data));
              double v364_data = s0_w0[243];
              tensorforge::intel_esimd::simd<double, 16> v366_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v366_data + (v288_data * v364_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run2 = tensorforge::slmLoad<double, 32>(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<double, 16> v369_data(glb_m1_run2.template select<16, 1>(0));
              double v370_data = s0_w0[4];
              tensorforge::intel_esimd::simd<double, 16> v372_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v372_data + (v369_data * v370_data));
              double v375_data = s0_w0[20];
              tensorforge::intel_esimd::simd<double, 16> v377_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v377_data + (v369_data * v375_data));
              double v380_data = s0_w0[36];
              tensorforge::intel_esimd::simd<double, 16> v382_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v382_data + (v369_data * v380_data));
              double v385_data = s0_w0[52];
              tensorforge::intel_esimd::simd<double, 16> v387_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v387_data + (v369_data * v385_data));
              double v390_data = s0_w0[68];
              tensorforge::intel_esimd::simd<double, 16> v392_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v392_data + (v369_data * v390_data));
              double v395_data = s0_w0[84];
              tensorforge::intel_esimd::simd<double, 16> v397_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v397_data + (v369_data * v395_data));
              double v400_data = s0_w0[100];
              tensorforge::intel_esimd::simd<double, 16> v402_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v402_data + (v369_data * v400_data));
              double v405_data = s0_w0[116];
              tensorforge::intel_esimd::simd<double, 16> v407_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v407_data + (v369_data * v405_data));
              double v410_data = s0_w0[132];
              tensorforge::intel_esimd::simd<double, 16> v412_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v412_data + (v369_data * v410_data));
              double v415_data = s0_w0[148];
              tensorforge::intel_esimd::simd<double, 16> v417_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v417_data + (v369_data * v415_data));
              double v420_data = s0_w0[164];
              tensorforge::intel_esimd::simd<double, 16> v422_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v422_data + (v369_data * v420_data));
              double v425_data = s0_w0[180];
              tensorforge::intel_esimd::simd<double, 16> v427_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v427_data + (v369_data * v425_data));
              double v430_data = s0_w0[196];
              tensorforge::intel_esimd::simd<double, 16> v432_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v432_data + (v369_data * v430_data));
              double v435_data = s0_w0[212];
              tensorforge::intel_esimd::simd<double, 16> v437_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v437_data + (v369_data * v435_data));
              double v440_data = s0_w0[228];
              tensorforge::intel_esimd::simd<double, 16> v442_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v442_data + (v369_data * v440_data));
              double v445_data = s0_w0[244];
              tensorforge::intel_esimd::simd<double, 16> v447_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v447_data + (v369_data * v445_data));
              tensorforge::intel_esimd::simd<double, 16> v450_data(glb_m1_run2.template select<16, 1>(16));
              double v451_data = s0_w0[5];
              tensorforge::intel_esimd::simd<double, 16> v453_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v453_data + (v450_data * v451_data));
              double v456_data = s0_w0[21];
              tensorforge::intel_esimd::simd<double, 16> v458_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v458_data + (v450_data * v456_data));
              double v461_data = s0_w0[37];
              tensorforge::intel_esimd::simd<double, 16> v463_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v463_data + (v450_data * v461_data));
              double v466_data = s0_w0[53];
              tensorforge::intel_esimd::simd<double, 16> v468_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v468_data + (v450_data * v466_data));
              double v471_data = s0_w0[69];
              tensorforge::intel_esimd::simd<double, 16> v473_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v473_data + (v450_data * v471_data));
              double v476_data = s0_w0[85];
              tensorforge::intel_esimd::simd<double, 16> v478_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v478_data + (v450_data * v476_data));
              double v481_data = s0_w0[101];
              tensorforge::intel_esimd::simd<double, 16> v483_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v483_data + (v450_data * v481_data));
              double v486_data = s0_w0[117];
              tensorforge::intel_esimd::simd<double, 16> v488_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v488_data + (v450_data * v486_data));
              double v491_data = s0_w0[133];
              tensorforge::intel_esimd::simd<double, 16> v493_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v493_data + (v450_data * v491_data));
              double v496_data = s0_w0[149];
              tensorforge::intel_esimd::simd<double, 16> v498_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v498_data + (v450_data * v496_data));
              double v501_data = s0_w0[165];
              tensorforge::intel_esimd::simd<double, 16> v503_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v503_data + (v450_data * v501_data));
              double v506_data = s0_w0[181];
              tensorforge::intel_esimd::simd<double, 16> v508_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v508_data + (v450_data * v506_data));
              double v511_data = s0_w0[197];
              tensorforge::intel_esimd::simd<double, 16> v513_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v513_data + (v450_data * v511_data));
              double v516_data = s0_w0[213];
              tensorforge::intel_esimd::simd<double, 16> v518_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v518_data + (v450_data * v516_data));
              double v521_data = s0_w0[229];
              tensorforge::intel_esimd::simd<double, 16> v523_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v523_data + (v450_data * v521_data));
              double v526_data = s0_w0[245];
              tensorforge::intel_esimd::simd<double, 16> v528_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v528_data + (v450_data * v526_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run3 = tensorforge::slmLoad<double, 32>(glb_m1 + (96_i32));
              tensorforge::intel_esimd::simd<double, 16> v531_data(glb_m1_run3.template select<16, 1>(0));
              double v532_data = s0_w0[6];
              tensorforge::intel_esimd::simd<double, 16> v534_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v534_data + (v531_data * v532_data));
              double v537_data = s0_w0[22];
              tensorforge::intel_esimd::simd<double, 16> v539_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v539_data + (v531_data * v537_data));
              double v542_data = s0_w0[38];
              tensorforge::intel_esimd::simd<double, 16> v544_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v544_data + (v531_data * v542_data));
              double v547_data = s0_w0[54];
              tensorforge::intel_esimd::simd<double, 16> v549_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v549_data + (v531_data * v547_data));
              double v552_data = s0_w0[70];
              tensorforge::intel_esimd::simd<double, 16> v554_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v554_data + (v531_data * v552_data));
              double v557_data = s0_w0[86];
              tensorforge::intel_esimd::simd<double, 16> v559_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v559_data + (v531_data * v557_data));
              double v562_data = s0_w0[102];
              tensorforge::intel_esimd::simd<double, 16> v564_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v564_data + (v531_data * v562_data));
              double v567_data = s0_w0[118];
              tensorforge::intel_esimd::simd<double, 16> v569_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v569_data + (v531_data * v567_data));
              double v572_data = s0_w0[134];
              tensorforge::intel_esimd::simd<double, 16> v574_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v574_data + (v531_data * v572_data));
              double v577_data = s0_w0[150];
              tensorforge::intel_esimd::simd<double, 16> v579_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v579_data + (v531_data * v577_data));
              double v582_data = s0_w0[166];
              tensorforge::intel_esimd::simd<double, 16> v584_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v584_data + (v531_data * v582_data));
              double v587_data = s0_w0[182];
              tensorforge::intel_esimd::simd<double, 16> v589_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v589_data + (v531_data * v587_data));
              double v592_data = s0_w0[198];
              tensorforge::intel_esimd::simd<double, 16> v594_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v594_data + (v531_data * v592_data));
              double v597_data = s0_w0[214];
              tensorforge::intel_esimd::simd<double, 16> v599_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v599_data + (v531_data * v597_data));
              double v602_data = s0_w0[230];
              tensorforge::intel_esimd::simd<double, 16> v604_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v604_data + (v531_data * v602_data));
              double v607_data = s0_w0[246];
              tensorforge::intel_esimd::simd<double, 16> v609_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v609_data + (v531_data * v607_data));
              tensorforge::intel_esimd::simd<double, 16> v612_data(glb_m1_run3.template select<16, 1>(16));
              double v613_data = s0_w0[7];
              tensorforge::intel_esimd::simd<double, 16> v615_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v615_data + (v612_data * v613_data));
              double v618_data = s0_w0[23];
              tensorforge::intel_esimd::simd<double, 16> v620_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v620_data + (v612_data * v618_data));
              double v623_data = s0_w0[39];
              tensorforge::intel_esimd::simd<double, 16> v625_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v625_data + (v612_data * v623_data));
              double v628_data = s0_w0[55];
              tensorforge::intel_esimd::simd<double, 16> v630_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v630_data + (v612_data * v628_data));
              double v633_data = s0_w0[71];
              tensorforge::intel_esimd::simd<double, 16> v635_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v635_data + (v612_data * v633_data));
              double v638_data = s0_w0[87];
              tensorforge::intel_esimd::simd<double, 16> v640_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v640_data + (v612_data * v638_data));
              double v643_data = s0_w0[103];
              tensorforge::intel_esimd::simd<double, 16> v645_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v645_data + (v612_data * v643_data));
              double v648_data = s0_w0[119];
              tensorforge::intel_esimd::simd<double, 16> v650_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v650_data + (v612_data * v648_data));
              double v653_data = s0_w0[135];
              tensorforge::intel_esimd::simd<double, 16> v655_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v655_data + (v612_data * v653_data));
              double v658_data = s0_w0[151];
              tensorforge::intel_esimd::simd<double, 16> v660_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v660_data + (v612_data * v658_data));
              double v663_data = s0_w0[167];
              tensorforge::intel_esimd::simd<double, 16> v665_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v665_data + (v612_data * v663_data));
              double v668_data = s0_w0[183];
              tensorforge::intel_esimd::simd<double, 16> v670_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v670_data + (v612_data * v668_data));
              double v673_data = s0_w0[199];
              tensorforge::intel_esimd::simd<double, 16> v675_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v675_data + (v612_data * v673_data));
              double v678_data = s0_w0[215];
              tensorforge::intel_esimd::simd<double, 16> v680_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v680_data + (v612_data * v678_data));
              double v683_data = s0_w0[231];
              tensorforge::intel_esimd::simd<double, 16> v685_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v685_data + (v612_data * v683_data));
              double v688_data = s0_w0[247];
              tensorforge::intel_esimd::simd<double, 16> v690_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v690_data + (v612_data * v688_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run4 = tensorforge::slmLoad<double, 32>(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<double, 16> v693_data(glb_m1_run4.template select<16, 1>(0));
              double v694_data = s0_w0[8];
              tensorforge::intel_esimd::simd<double, 16> v696_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v696_data + (v693_data * v694_data));
              double v699_data = s0_w0[24];
              tensorforge::intel_esimd::simd<double, 16> v701_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v701_data + (v693_data * v699_data));
              double v704_data = s0_w0[40];
              tensorforge::intel_esimd::simd<double, 16> v706_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v706_data + (v693_data * v704_data));
              double v709_data = s0_w0[56];
              tensorforge::intel_esimd::simd<double, 16> v711_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v711_data + (v693_data * v709_data));
              double v714_data = s0_w0[72];
              tensorforge::intel_esimd::simd<double, 16> v716_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v716_data + (v693_data * v714_data));
              double v719_data = s0_w0[88];
              tensorforge::intel_esimd::simd<double, 16> v721_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v721_data + (v693_data * v719_data));
              double v724_data = s0_w0[104];
              tensorforge::intel_esimd::simd<double, 16> v726_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v726_data + (v693_data * v724_data));
              double v729_data = s0_w0[120];
              tensorforge::intel_esimd::simd<double, 16> v731_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v731_data + (v693_data * v729_data));
              double v734_data = s0_w0[136];
              tensorforge::intel_esimd::simd<double, 16> v736_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v736_data + (v693_data * v734_data));
              double v739_data = s0_w0[152];
              tensorforge::intel_esimd::simd<double, 16> v741_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v741_data + (v693_data * v739_data));
              double v744_data = s0_w0[168];
              tensorforge::intel_esimd::simd<double, 16> v746_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v746_data + (v693_data * v744_data));
              double v749_data = s0_w0[184];
              tensorforge::intel_esimd::simd<double, 16> v751_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v751_data + (v693_data * v749_data));
              double v754_data = s0_w0[200];
              tensorforge::intel_esimd::simd<double, 16> v756_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v756_data + (v693_data * v754_data));
              double v759_data = s0_w0[216];
              tensorforge::intel_esimd::simd<double, 16> v761_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v761_data + (v693_data * v759_data));
              double v764_data = s0_w0[232];
              tensorforge::intel_esimd::simd<double, 16> v766_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v766_data + (v693_data * v764_data));
              double v769_data = s0_w0[248];
              tensorforge::intel_esimd::simd<double, 16> v771_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v771_data + (v693_data * v769_data));
              tensorforge::intel_esimd::simd<double, 16> v774_data(glb_m1_run4.template select<16, 1>(16));
              double v775_data = s0_w0[9];
              tensorforge::intel_esimd::simd<double, 16> v777_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v777_data + (v774_data * v775_data));
              double v780_data = s0_w0[25];
              tensorforge::intel_esimd::simd<double, 16> v782_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v782_data + (v774_data * v780_data));
              double v785_data = s0_w0[41];
              tensorforge::intel_esimd::simd<double, 16> v787_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v787_data + (v774_data * v785_data));
              double v790_data = s0_w0[57];
              tensorforge::intel_esimd::simd<double, 16> v792_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v792_data + (v774_data * v790_data));
              double v795_data = s0_w0[73];
              tensorforge::intel_esimd::simd<double, 16> v797_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v797_data + (v774_data * v795_data));
              double v800_data = s0_w0[89];
              tensorforge::intel_esimd::simd<double, 16> v802_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v802_data + (v774_data * v800_data));
              double v805_data = s0_w0[105];
              tensorforge::intel_esimd::simd<double, 16> v807_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v807_data + (v774_data * v805_data));
              double v810_data = s0_w0[121];
              tensorforge::intel_esimd::simd<double, 16> v812_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v812_data + (v774_data * v810_data));
              double v815_data = s0_w0[137];
              tensorforge::intel_esimd::simd<double, 16> v817_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v817_data + (v774_data * v815_data));
              double v820_data = s0_w0[153];
              tensorforge::intel_esimd::simd<double, 16> v822_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v822_data + (v774_data * v820_data));
              double v825_data = s0_w0[169];
              tensorforge::intel_esimd::simd<double, 16> v827_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v827_data + (v774_data * v825_data));
              double v830_data = s0_w0[185];
              tensorforge::intel_esimd::simd<double, 16> v832_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v832_data + (v774_data * v830_data));
              double v835_data = s0_w0[201];
              tensorforge::intel_esimd::simd<double, 16> v837_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v837_data + (v774_data * v835_data));
              double v840_data = s0_w0[217];
              tensorforge::intel_esimd::simd<double, 16> v842_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v842_data + (v774_data * v840_data));
              double v845_data = s0_w0[233];
              tensorforge::intel_esimd::simd<double, 16> v847_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v847_data + (v774_data * v845_data));
              double v850_data = s0_w0[249];
              tensorforge::intel_esimd::simd<double, 16> v852_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v852_data + (v774_data * v850_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run5 = tensorforge::slmLoad<double, 32>(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<double, 16> v855_data(glb_m1_run5.template select<16, 1>(0));
              double v856_data = s0_w0[10];
              tensorforge::intel_esimd::simd<double, 16> v858_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v858_data + (v855_data * v856_data));
              double v861_data = s0_w0[26];
              tensorforge::intel_esimd::simd<double, 16> v863_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v863_data + (v855_data * v861_data));
              double v866_data = s0_w0[42];
              tensorforge::intel_esimd::simd<double, 16> v868_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v868_data + (v855_data * v866_data));
              double v871_data = s0_w0[58];
              tensorforge::intel_esimd::simd<double, 16> v873_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v873_data + (v855_data * v871_data));
              double v876_data = s0_w0[74];
              tensorforge::intel_esimd::simd<double, 16> v878_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v878_data + (v855_data * v876_data));
              double v881_data = s0_w0[90];
              tensorforge::intel_esimd::simd<double, 16> v883_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v883_data + (v855_data * v881_data));
              double v886_data = s0_w0[106];
              tensorforge::intel_esimd::simd<double, 16> v888_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v888_data + (v855_data * v886_data));
              double v891_data = s0_w0[122];
              tensorforge::intel_esimd::simd<double, 16> v893_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v893_data + (v855_data * v891_data));
              double v896_data = s0_w0[138];
              tensorforge::intel_esimd::simd<double, 16> v898_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v898_data + (v855_data * v896_data));
              double v901_data = s0_w0[154];
              tensorforge::intel_esimd::simd<double, 16> v903_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v903_data + (v855_data * v901_data));
              double v906_data = s0_w0[170];
              tensorforge::intel_esimd::simd<double, 16> v908_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v908_data + (v855_data * v906_data));
              double v911_data = s0_w0[186];
              tensorforge::intel_esimd::simd<double, 16> v913_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v913_data + (v855_data * v911_data));
              double v916_data = s0_w0[202];
              tensorforge::intel_esimd::simd<double, 16> v918_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v918_data + (v855_data * v916_data));
              double v921_data = s0_w0[218];
              tensorforge::intel_esimd::simd<double, 16> v923_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v923_data + (v855_data * v921_data));
              double v926_data = s0_w0[234];
              tensorforge::intel_esimd::simd<double, 16> v928_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v928_data + (v855_data * v926_data));
              double v931_data = s0_w0[250];
              tensorforge::intel_esimd::simd<double, 16> v933_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v933_data + (v855_data * v931_data));
              tensorforge::intel_esimd::simd<double, 16> v936_data(glb_m1_run5.template select<16, 1>(16));
              double v937_data = s0_w0[11];
              tensorforge::intel_esimd::simd<double, 16> v939_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v939_data + (v936_data * v937_data));
              double v942_data = s0_w0[27];
              tensorforge::intel_esimd::simd<double, 16> v944_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v944_data + (v936_data * v942_data));
              double v947_data = s0_w0[43];
              tensorforge::intel_esimd::simd<double, 16> v949_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v949_data + (v936_data * v947_data));
              double v952_data = s0_w0[59];
              tensorforge::intel_esimd::simd<double, 16> v954_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v954_data + (v936_data * v952_data));
              double v957_data = s0_w0[75];
              tensorforge::intel_esimd::simd<double, 16> v959_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v959_data + (v936_data * v957_data));
              double v962_data = s0_w0[91];
              tensorforge::intel_esimd::simd<double, 16> v964_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v964_data + (v936_data * v962_data));
              double v967_data = s0_w0[107];
              tensorforge::intel_esimd::simd<double, 16> v969_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v969_data + (v936_data * v967_data));
              double v972_data = s0_w0[123];
              tensorforge::intel_esimd::simd<double, 16> v974_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v974_data + (v936_data * v972_data));
              double v977_data = s0_w0[139];
              tensorforge::intel_esimd::simd<double, 16> v979_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v979_data + (v936_data * v977_data));
              double v982_data = s0_w0[155];
              tensorforge::intel_esimd::simd<double, 16> v984_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v984_data + (v936_data * v982_data));
              double v987_data = s0_w0[171];
              tensorforge::intel_esimd::simd<double, 16> v989_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v989_data + (v936_data * v987_data));
              double v992_data = s0_w0[187];
              tensorforge::intel_esimd::simd<double, 16> v994_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v994_data + (v936_data * v992_data));
              double v997_data = s0_w0[203];
              tensorforge::intel_esimd::simd<double, 16> v999_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v999_data + (v936_data * v997_data));
              double v1002_data = s0_w0[219];
              tensorforge::intel_esimd::simd<double, 16> v1004_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1004_data + (v936_data * v1002_data));
              double v1007_data = s0_w0[235];
              tensorforge::intel_esimd::simd<double, 16> v1009_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1009_data + (v936_data * v1007_data));
              double v1012_data = s0_w0[251];
              tensorforge::intel_esimd::simd<double, 16> v1014_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1014_data + (v936_data * v1012_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run6 = tensorforge::slmLoad<double, 32>(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<double, 16> v1017_data(glb_m1_run6.template select<16, 1>(0));
              double v1018_data = s0_w0[12];
              tensorforge::intel_esimd::simd<double, 16> v1020_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v1020_data + (v1017_data * v1018_data));
              double v1023_data = s0_w0[28];
              tensorforge::intel_esimd::simd<double, 16> v1025_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v1025_data + (v1017_data * v1023_data));
              double v1028_data = s0_w0[44];
              tensorforge::intel_esimd::simd<double, 16> v1030_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v1030_data + (v1017_data * v1028_data));
              double v1033_data = s0_w0[60];
              tensorforge::intel_esimd::simd<double, 16> v1035_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v1035_data + (v1017_data * v1033_data));
              double v1038_data = s0_w0[76];
              tensorforge::intel_esimd::simd<double, 16> v1040_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v1040_data + (v1017_data * v1038_data));
              double v1043_data = s0_w0[92];
              tensorforge::intel_esimd::simd<double, 16> v1045_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v1045_data + (v1017_data * v1043_data));
              double v1048_data = s0_w0[108];
              tensorforge::intel_esimd::simd<double, 16> v1050_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v1050_data + (v1017_data * v1048_data));
              double v1053_data = s0_w0[124];
              tensorforge::intel_esimd::simd<double, 16> v1055_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v1055_data + (v1017_data * v1053_data));
              double v1058_data = s0_w0[140];
              tensorforge::intel_esimd::simd<double, 16> v1060_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v1060_data + (v1017_data * v1058_data));
              double v1063_data = s0_w0[156];
              tensorforge::intel_esimd::simd<double, 16> v1065_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v1065_data + (v1017_data * v1063_data));
              double v1068_data = s0_w0[172];
              tensorforge::intel_esimd::simd<double, 16> v1070_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v1070_data + (v1017_data * v1068_data));
              double v1073_data = s0_w0[188];
              tensorforge::intel_esimd::simd<double, 16> v1075_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v1075_data + (v1017_data * v1073_data));
              double v1078_data = s0_w0[204];
              tensorforge::intel_esimd::simd<double, 16> v1080_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v1080_data + (v1017_data * v1078_data));
              double v1083_data = s0_w0[220];
              tensorforge::intel_esimd::simd<double, 16> v1085_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1085_data + (v1017_data * v1083_data));
              double v1088_data = s0_w0[236];
              tensorforge::intel_esimd::simd<double, 16> v1090_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1090_data + (v1017_data * v1088_data));
              double v1093_data = s0_w0[252];
              tensorforge::intel_esimd::simd<double, 16> v1095_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1095_data + (v1017_data * v1093_data));
              tensorforge::intel_esimd::simd<double, 16> v1098_data(glb_m1_run6.template select<16, 1>(16));
              double v1099_data = s0_w0[13];
              tensorforge::intel_esimd::simd<double, 16> v1101_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v1101_data + (v1098_data * v1099_data));
              double v1104_data = s0_w0[29];
              tensorforge::intel_esimd::simd<double, 16> v1106_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v1106_data + (v1098_data * v1104_data));
              double v1109_data = s0_w0[45];
              tensorforge::intel_esimd::simd<double, 16> v1111_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v1111_data + (v1098_data * v1109_data));
              double v1114_data = s0_w0[61];
              tensorforge::intel_esimd::simd<double, 16> v1116_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v1116_data + (v1098_data * v1114_data));
              double v1119_data = s0_w0[77];
              tensorforge::intel_esimd::simd<double, 16> v1121_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v1121_data + (v1098_data * v1119_data));
              double v1124_data = s0_w0[93];
              tensorforge::intel_esimd::simd<double, 16> v1126_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v1126_data + (v1098_data * v1124_data));
              double v1129_data = s0_w0[109];
              tensorforge::intel_esimd::simd<double, 16> v1131_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v1131_data + (v1098_data * v1129_data));
              double v1134_data = s0_w0[125];
              tensorforge::intel_esimd::simd<double, 16> v1136_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v1136_data + (v1098_data * v1134_data));
              double v1139_data = s0_w0[141];
              tensorforge::intel_esimd::simd<double, 16> v1141_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v1141_data + (v1098_data * v1139_data));
              double v1144_data = s0_w0[157];
              tensorforge::intel_esimd::simd<double, 16> v1146_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v1146_data + (v1098_data * v1144_data));
              double v1149_data = s0_w0[173];
              tensorforge::intel_esimd::simd<double, 16> v1151_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v1151_data + (v1098_data * v1149_data));
              double v1154_data = s0_w0[189];
              tensorforge::intel_esimd::simd<double, 16> v1156_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v1156_data + (v1098_data * v1154_data));
              double v1159_data = s0_w0[205];
              tensorforge::intel_esimd::simd<double, 16> v1161_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v1161_data + (v1098_data * v1159_data));
              double v1164_data = s0_w0[221];
              tensorforge::intel_esimd::simd<double, 16> v1166_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1166_data + (v1098_data * v1164_data));
              double v1169_data = s0_w0[237];
              tensorforge::intel_esimd::simd<double, 16> v1171_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1171_data + (v1098_data * v1169_data));
              double v1174_data = s0_w0[253];
              tensorforge::intel_esimd::simd<double, 16> v1176_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1176_data + (v1098_data * v1174_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run7 = tensorforge::slmLoad<double, 32>(glb_m1 + (224_i32));
              tensorforge::intel_esimd::simd<double, 16> v1179_data(glb_m1_run7.template select<16, 1>(0));
              double v1180_data = s0_w0[14];
              tensorforge::intel_esimd::simd<double, 16> v1182_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v1182_data + (v1179_data * v1180_data));
              double v1185_data = s0_w0[30];
              tensorforge::intel_esimd::simd<double, 16> v1187_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v1187_data + (v1179_data * v1185_data));
              double v1190_data = s0_w0[46];
              tensorforge::intel_esimd::simd<double, 16> v1192_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v1192_data + (v1179_data * v1190_data));
              double v1195_data = s0_w0[62];
              tensorforge::intel_esimd::simd<double, 16> v1197_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v1197_data + (v1179_data * v1195_data));
              double v1200_data = s0_w0[78];
              tensorforge::intel_esimd::simd<double, 16> v1202_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v1202_data + (v1179_data * v1200_data));
              double v1205_data = s0_w0[94];
              tensorforge::intel_esimd::simd<double, 16> v1207_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v1207_data + (v1179_data * v1205_data));
              double v1210_data = s0_w0[110];
              tensorforge::intel_esimd::simd<double, 16> v1212_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v1212_data + (v1179_data * v1210_data));
              double v1215_data = s0_w0[126];
              tensorforge::intel_esimd::simd<double, 16> v1217_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v1217_data + (v1179_data * v1215_data));
              double v1220_data = s0_w0[142];
              tensorforge::intel_esimd::simd<double, 16> v1222_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v1222_data + (v1179_data * v1220_data));
              double v1225_data = s0_w0[158];
              tensorforge::intel_esimd::simd<double, 16> v1227_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v1227_data + (v1179_data * v1225_data));
              double v1230_data = s0_w0[174];
              tensorforge::intel_esimd::simd<double, 16> v1232_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v1232_data + (v1179_data * v1230_data));
              double v1235_data = s0_w0[190];
              tensorforge::intel_esimd::simd<double, 16> v1237_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v1237_data + (v1179_data * v1235_data));
              double v1240_data = s0_w0[206];
              tensorforge::intel_esimd::simd<double, 16> v1242_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v1242_data + (v1179_data * v1240_data));
              double v1245_data = s0_w0[222];
              tensorforge::intel_esimd::simd<double, 16> v1247_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1247_data + (v1179_data * v1245_data));
              double v1250_data = s0_w0[238];
              tensorforge::intel_esimd::simd<double, 16> v1252_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1252_data + (v1179_data * v1250_data));
              double v1255_data = s0_w0[254];
              tensorforge::intel_esimd::simd<double, 16> v1257_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1257_data + (v1179_data * v1255_data));
              tensorforge::intel_esimd::simd<double, 16> v1260_data(glb_m1_run7.template select<16, 1>(16));
              double v1261_data = s0_w0[15];
              tensorforge::intel_esimd::simd<double, 16> v1263_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v1263_data + (v1260_data * v1261_data));
              double v1266_data = s0_w0[31];
              tensorforge::intel_esimd::simd<double, 16> v1268_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v1268_data + (v1260_data * v1266_data));
              double v1271_data = s0_w0[47];
              tensorforge::intel_esimd::simd<double, 16> v1273_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v1273_data + (v1260_data * v1271_data));
              double v1276_data = s0_w0[63];
              tensorforge::intel_esimd::simd<double, 16> v1278_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v1278_data + (v1260_data * v1276_data));
              double v1281_data = s0_w0[79];
              tensorforge::intel_esimd::simd<double, 16> v1283_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v1283_data + (v1260_data * v1281_data));
              double v1286_data = s0_w0[95];
              tensorforge::intel_esimd::simd<double, 16> v1288_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v1288_data + (v1260_data * v1286_data));
              double v1291_data = s0_w0[111];
              tensorforge::intel_esimd::simd<double, 16> v1293_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v1293_data + (v1260_data * v1291_data));
              double v1296_data = s0_w0[127];
              tensorforge::intel_esimd::simd<double, 16> v1298_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v1298_data + (v1260_data * v1296_data));
              double v1301_data = s0_w0[143];
              tensorforge::intel_esimd::simd<double, 16> v1303_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v1303_data + (v1260_data * v1301_data));
              double v1306_data = s0_w0[159];
              tensorforge::intel_esimd::simd<double, 16> v1308_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v1308_data + (v1260_data * v1306_data));
              double v1311_data = s0_w0[175];
              tensorforge::intel_esimd::simd<double, 16> v1313_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v1313_data + (v1260_data * v1311_data));
              double v1316_data = s0_w0[191];
              tensorforge::intel_esimd::simd<double, 16> v1318_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v1318_data + (v1260_data * v1316_data));
              double v1321_data = s0_w0[207];
              tensorforge::intel_esimd::simd<double, 16> v1323_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v1323_data + (v1260_data * v1321_data));
              double v1326_data = s0_w0[223];
              tensorforge::intel_esimd::simd<double, 16> v1328_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1328_data + (v1260_data * v1326_data));
              double v1331_data = s0_w0[239];
              tensorforge::intel_esimd::simd<double, 16> v1333_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1333_data + (v1260_data * v1331_data));
              double v1336_data = s0_w0[255];
              tensorforge::intel_esimd::simd<double, 16> v1338_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1338_data + (v1260_data * v1336_data));
              // r0 = ir0
              #pragma unroll
              for (int32_t v1340_n0 = 0; v1340_n0 < 1; ++v1340_n0) {
                int32_t v1342_a = v1340_n0 * 16;
                #pragma unroll
                for (int32_t v1341_n1 = 0; v1341_n1 < 16; ++v1341_n1) {
                  int32_t v1344_a = v1342_a + (v1341_n1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v1345_data(ir0.template select<16, 1>(v1344_a));
                  r0.template select<16, 1>(v1344_a) = v1345_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v1346_i0 = 0; v1346_i0 < 1; ++v1346_i0) {
                int32_t v1348_a = v1346_i0 * 16;
                #pragma unroll
                for (int32_t v1347_i1 = 0; v1347_i1 < 16; ++v1347_i1) {
                  int32_t v1350_a = v1348_a + (v1347_i1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v1351_data(r0.template select<16, 1>(v1350_a));
                  v1351_data.copy_to(glb_m0 + (v1350_a));
                }
              }
            }
          }
        }
      }
    });
  });
}

