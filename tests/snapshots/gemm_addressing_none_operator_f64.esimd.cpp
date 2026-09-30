// === base name ===
kernel_87c4da6a6f71cf54

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_87c4da6a6f71cf54 = {{1, 32, 1}, 16, 16, 1, 32, 71680, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_87c4da6a6f71cf54(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_87c4da6a6f71cf54(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_87c4da6a6f71cf54(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_87c4da6a6f71cf54(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_87c4da6a6f71cf54(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_87c4da6a6f71cf54(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_87c4da6a6f71cf54(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<8960 * sizeof(double)>(); {
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
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<double> totalShrMem = tensorforge::SlmPtr<double>(0);
          tensorforge::SlmPtr<double> localShrMem0 = totalShrMem + (272 * item.get_local_id(1) + 256);
          tensorforge::SlmPtr<double> tempShrMem = localShrMem0 + (256);
          const double *const __restrict__ ptr_glb_m1 = &m1[0];
          tensorforge::SlmPtr<double> glb_m1 = totalShrMem + (0);
          // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<double, 16> v5_ld;
            v5_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v5_ld);
          }
          if (item.get_local_id(1) == 1) {
            tensorforge::intel_esimd::simd<double, 16> v6_ld;
            v6_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v6_ld);
          }
          if (item.get_local_id(1) == 2) {
            tensorforge::intel_esimd::simd<double, 16> v7_ld;
            v7_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v7_ld);
          }
          if (item.get_local_id(1) == 3) {
            tensorforge::intel_esimd::simd<double, 16> v8_ld;
            v8_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v8_ld);
          }
          if (item.get_local_id(1) == 4) {
            tensorforge::intel_esimd::simd<double, 16> v9_ld;
            v9_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v9_ld);
          }
          if (item.get_local_id(1) == 5) {
            tensorforge::intel_esimd::simd<double, 16> v10_ld;
            v10_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v10_ld);
          }
          if (item.get_local_id(1) == 6) {
            tensorforge::intel_esimd::simd<double, 16> v11_ld;
            v11_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v11_ld);
          }
          if (item.get_local_id(1) == 7) {
            tensorforge::intel_esimd::simd<double, 16> v12_ld;
            v12_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v12_ld);
          }
          if (item.get_local_id(1) == 8) {
            tensorforge::intel_esimd::simd<double, 16> v13_ld;
            v13_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v13_ld);
          }
          if (item.get_local_id(1) == 9) {
            tensorforge::intel_esimd::simd<double, 16> v14_ld;
            v14_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v14_ld);
          }
          if (item.get_local_id(1) == 10) {
            tensorforge::intel_esimd::simd<double, 16> v15_ld;
            v15_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v15_ld);
          }
          if (item.get_local_id(1) == 11) {
            tensorforge::intel_esimd::simd<double, 16> v16_ld;
            v16_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v16_ld);
          }
          if (item.get_local_id(1) == 12) {
            tensorforge::intel_esimd::simd<double, 16> v17_ld;
            v17_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v17_ld);
          }
          if (item.get_local_id(1) == 13) {
            tensorforge::intel_esimd::simd<double, 16> v18_ld;
            v18_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v18_ld);
          }
          if (item.get_local_id(1) == 14) {
            tensorforge::intel_esimd::simd<double, 16> v19_ld;
            v19_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v19_ld);
          }
          if (item.get_local_id(1) == 15) {
            tensorforge::intel_esimd::simd<double, 16> v20_ld;
            v20_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0));
            tensorforge::slmStore<double, 16>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 16) + 0), v20_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
          item.barrier();
          tensorforge::SlmPtr<double> s0 = localShrMem0 + (0);
          for (size_t v23_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v23_batchId0 < numElements0; v23_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v24_ahead1 = v23_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v26_batchId1 = (v24_ahead1 < numElements0) ? v24_ahead1 : v23_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v23_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v23_batchId0 * 256 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v23_batchId0 * 256 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              #pragma unroll
              for (int32_t i = 0; i < 16; i += 2) {
                tensorforge::intel_esimd::simd<double, 32> v33_ld;
                v33_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + i * 16));
                tensorforge::slmStore<double, 32>(s0 + (0 + 0 + 2 * 0 + i * 16), v33_ld);
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<double, 256> r0(0.0);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<double, 256> ir0(0.0);
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run0 = tensorforge::slmLoad<double, 32>(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<double, 16> v39_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<double, 256> s0_w0 = tensorforge::slmLoad<double, 256>(s0 + 0);
              double v40_data = s0_w0[0];
              tensorforge::intel_esimd::simd<double, 16> v42_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v42_data + (v39_data * v40_data));
              double v45_data = s0_w0[16];
              tensorforge::intel_esimd::simd<double, 16> v47_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v47_data + (v39_data * v45_data));
              double v50_data = s0_w0[32];
              tensorforge::intel_esimd::simd<double, 16> v52_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v52_data + (v39_data * v50_data));
              double v55_data = s0_w0[48];
              tensorforge::intel_esimd::simd<double, 16> v57_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v57_data + (v39_data * v55_data));
              double v60_data = s0_w0[64];
              tensorforge::intel_esimd::simd<double, 16> v62_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v62_data + (v39_data * v60_data));
              double v65_data = s0_w0[80];
              tensorforge::intel_esimd::simd<double, 16> v67_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v67_data + (v39_data * v65_data));
              double v70_data = s0_w0[96];
              tensorforge::intel_esimd::simd<double, 16> v72_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v72_data + (v39_data * v70_data));
              double v75_data = s0_w0[112];
              tensorforge::intel_esimd::simd<double, 16> v77_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v77_data + (v39_data * v75_data));
              double v80_data = s0_w0[128];
              tensorforge::intel_esimd::simd<double, 16> v82_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v82_data + (v39_data * v80_data));
              double v85_data = s0_w0[144];
              tensorforge::intel_esimd::simd<double, 16> v87_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v87_data + (v39_data * v85_data));
              double v90_data = s0_w0[160];
              tensorforge::intel_esimd::simd<double, 16> v92_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v92_data + (v39_data * v90_data));
              double v95_data = s0_w0[176];
              tensorforge::intel_esimd::simd<double, 16> v97_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v97_data + (v39_data * v95_data));
              double v100_data = s0_w0[192];
              tensorforge::intel_esimd::simd<double, 16> v102_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v102_data + (v39_data * v100_data));
              double v105_data = s0_w0[208];
              tensorforge::intel_esimd::simd<double, 16> v107_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v107_data + (v39_data * v105_data));
              double v110_data = s0_w0[224];
              tensorforge::intel_esimd::simd<double, 16> v112_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v112_data + (v39_data * v110_data));
              double v115_data = s0_w0[240];
              tensorforge::intel_esimd::simd<double, 16> v117_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v117_data + (v39_data * v115_data));
              tensorforge::intel_esimd::simd<double, 16> v120_data(glb_m1_run0.template select<16, 1>(16));
              double v121_data = s0_w0[1];
              tensorforge::intel_esimd::simd<double, 16> v123_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v123_data + (v120_data * v121_data));
              double v126_data = s0_w0[17];
              tensorforge::intel_esimd::simd<double, 16> v128_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v128_data + (v120_data * v126_data));
              double v131_data = s0_w0[33];
              tensorforge::intel_esimd::simd<double, 16> v133_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v133_data + (v120_data * v131_data));
              double v136_data = s0_w0[49];
              tensorforge::intel_esimd::simd<double, 16> v138_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v138_data + (v120_data * v136_data));
              double v141_data = s0_w0[65];
              tensorforge::intel_esimd::simd<double, 16> v143_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v143_data + (v120_data * v141_data));
              double v146_data = s0_w0[81];
              tensorforge::intel_esimd::simd<double, 16> v148_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v148_data + (v120_data * v146_data));
              double v151_data = s0_w0[97];
              tensorforge::intel_esimd::simd<double, 16> v153_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v153_data + (v120_data * v151_data));
              double v156_data = s0_w0[113];
              tensorforge::intel_esimd::simd<double, 16> v158_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v158_data + (v120_data * v156_data));
              double v161_data = s0_w0[129];
              tensorforge::intel_esimd::simd<double, 16> v163_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v163_data + (v120_data * v161_data));
              double v166_data = s0_w0[145];
              tensorforge::intel_esimd::simd<double, 16> v168_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v168_data + (v120_data * v166_data));
              double v171_data = s0_w0[161];
              tensorforge::intel_esimd::simd<double, 16> v173_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v173_data + (v120_data * v171_data));
              double v176_data = s0_w0[177];
              tensorforge::intel_esimd::simd<double, 16> v178_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v178_data + (v120_data * v176_data));
              double v181_data = s0_w0[193];
              tensorforge::intel_esimd::simd<double, 16> v183_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v183_data + (v120_data * v181_data));
              double v186_data = s0_w0[209];
              tensorforge::intel_esimd::simd<double, 16> v188_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v188_data + (v120_data * v186_data));
              double v191_data = s0_w0[225];
              tensorforge::intel_esimd::simd<double, 16> v193_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v193_data + (v120_data * v191_data));
              double v196_data = s0_w0[241];
              tensorforge::intel_esimd::simd<double, 16> v198_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v198_data + (v120_data * v196_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run1 = tensorforge::slmLoad<double, 32>(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<double, 16> v201_data(glb_m1_run1.template select<16, 1>(0));
              double v202_data = s0_w0[2];
              tensorforge::intel_esimd::simd<double, 16> v204_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v204_data + (v201_data * v202_data));
              double v207_data = s0_w0[18];
              tensorforge::intel_esimd::simd<double, 16> v209_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v209_data + (v201_data * v207_data));
              double v212_data = s0_w0[34];
              tensorforge::intel_esimd::simd<double, 16> v214_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v214_data + (v201_data * v212_data));
              double v217_data = s0_w0[50];
              tensorforge::intel_esimd::simd<double, 16> v219_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v219_data + (v201_data * v217_data));
              double v222_data = s0_w0[66];
              tensorforge::intel_esimd::simd<double, 16> v224_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v224_data + (v201_data * v222_data));
              double v227_data = s0_w0[82];
              tensorforge::intel_esimd::simd<double, 16> v229_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v229_data + (v201_data * v227_data));
              double v232_data = s0_w0[98];
              tensorforge::intel_esimd::simd<double, 16> v234_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v234_data + (v201_data * v232_data));
              double v237_data = s0_w0[114];
              tensorforge::intel_esimd::simd<double, 16> v239_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v239_data + (v201_data * v237_data));
              double v242_data = s0_w0[130];
              tensorforge::intel_esimd::simd<double, 16> v244_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v244_data + (v201_data * v242_data));
              double v247_data = s0_w0[146];
              tensorforge::intel_esimd::simd<double, 16> v249_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v249_data + (v201_data * v247_data));
              double v252_data = s0_w0[162];
              tensorforge::intel_esimd::simd<double, 16> v254_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v254_data + (v201_data * v252_data));
              double v257_data = s0_w0[178];
              tensorforge::intel_esimd::simd<double, 16> v259_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v259_data + (v201_data * v257_data));
              double v262_data = s0_w0[194];
              tensorforge::intel_esimd::simd<double, 16> v264_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v264_data + (v201_data * v262_data));
              double v267_data = s0_w0[210];
              tensorforge::intel_esimd::simd<double, 16> v269_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v269_data + (v201_data * v267_data));
              double v272_data = s0_w0[226];
              tensorforge::intel_esimd::simd<double, 16> v274_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v274_data + (v201_data * v272_data));
              double v277_data = s0_w0[242];
              tensorforge::intel_esimd::simd<double, 16> v279_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v279_data + (v201_data * v277_data));
              tensorforge::intel_esimd::simd<double, 16> v282_data(glb_m1_run1.template select<16, 1>(16));
              double v283_data = s0_w0[3];
              tensorforge::intel_esimd::simd<double, 16> v285_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v285_data + (v282_data * v283_data));
              double v288_data = s0_w0[19];
              tensorforge::intel_esimd::simd<double, 16> v290_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v290_data + (v282_data * v288_data));
              double v293_data = s0_w0[35];
              tensorforge::intel_esimd::simd<double, 16> v295_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v295_data + (v282_data * v293_data));
              double v298_data = s0_w0[51];
              tensorforge::intel_esimd::simd<double, 16> v300_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v300_data + (v282_data * v298_data));
              double v303_data = s0_w0[67];
              tensorforge::intel_esimd::simd<double, 16> v305_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v305_data + (v282_data * v303_data));
              double v308_data = s0_w0[83];
              tensorforge::intel_esimd::simd<double, 16> v310_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v310_data + (v282_data * v308_data));
              double v313_data = s0_w0[99];
              tensorforge::intel_esimd::simd<double, 16> v315_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v315_data + (v282_data * v313_data));
              double v318_data = s0_w0[115];
              tensorforge::intel_esimd::simd<double, 16> v320_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v320_data + (v282_data * v318_data));
              double v323_data = s0_w0[131];
              tensorforge::intel_esimd::simd<double, 16> v325_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v325_data + (v282_data * v323_data));
              double v328_data = s0_w0[147];
              tensorforge::intel_esimd::simd<double, 16> v330_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v330_data + (v282_data * v328_data));
              double v333_data = s0_w0[163];
              tensorforge::intel_esimd::simd<double, 16> v335_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v335_data + (v282_data * v333_data));
              double v338_data = s0_w0[179];
              tensorforge::intel_esimd::simd<double, 16> v340_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v340_data + (v282_data * v338_data));
              double v343_data = s0_w0[195];
              tensorforge::intel_esimd::simd<double, 16> v345_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v345_data + (v282_data * v343_data));
              double v348_data = s0_w0[211];
              tensorforge::intel_esimd::simd<double, 16> v350_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v350_data + (v282_data * v348_data));
              double v353_data = s0_w0[227];
              tensorforge::intel_esimd::simd<double, 16> v355_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v355_data + (v282_data * v353_data));
              double v358_data = s0_w0[243];
              tensorforge::intel_esimd::simd<double, 16> v360_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v360_data + (v282_data * v358_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run2 = tensorforge::slmLoad<double, 32>(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<double, 16> v363_data(glb_m1_run2.template select<16, 1>(0));
              double v364_data = s0_w0[4];
              tensorforge::intel_esimd::simd<double, 16> v366_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v366_data + (v363_data * v364_data));
              double v369_data = s0_w0[20];
              tensorforge::intel_esimd::simd<double, 16> v371_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v371_data + (v363_data * v369_data));
              double v374_data = s0_w0[36];
              tensorforge::intel_esimd::simd<double, 16> v376_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v376_data + (v363_data * v374_data));
              double v379_data = s0_w0[52];
              tensorforge::intel_esimd::simd<double, 16> v381_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v381_data + (v363_data * v379_data));
              double v384_data = s0_w0[68];
              tensorforge::intel_esimd::simd<double, 16> v386_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v386_data + (v363_data * v384_data));
              double v389_data = s0_w0[84];
              tensorforge::intel_esimd::simd<double, 16> v391_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v391_data + (v363_data * v389_data));
              double v394_data = s0_w0[100];
              tensorforge::intel_esimd::simd<double, 16> v396_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v396_data + (v363_data * v394_data));
              double v399_data = s0_w0[116];
              tensorforge::intel_esimd::simd<double, 16> v401_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v401_data + (v363_data * v399_data));
              double v404_data = s0_w0[132];
              tensorforge::intel_esimd::simd<double, 16> v406_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v406_data + (v363_data * v404_data));
              double v409_data = s0_w0[148];
              tensorforge::intel_esimd::simd<double, 16> v411_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v411_data + (v363_data * v409_data));
              double v414_data = s0_w0[164];
              tensorforge::intel_esimd::simd<double, 16> v416_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v416_data + (v363_data * v414_data));
              double v419_data = s0_w0[180];
              tensorforge::intel_esimd::simd<double, 16> v421_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v421_data + (v363_data * v419_data));
              double v424_data = s0_w0[196];
              tensorforge::intel_esimd::simd<double, 16> v426_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v426_data + (v363_data * v424_data));
              double v429_data = s0_w0[212];
              tensorforge::intel_esimd::simd<double, 16> v431_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v431_data + (v363_data * v429_data));
              double v434_data = s0_w0[228];
              tensorforge::intel_esimd::simd<double, 16> v436_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v436_data + (v363_data * v434_data));
              double v439_data = s0_w0[244];
              tensorforge::intel_esimd::simd<double, 16> v441_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v441_data + (v363_data * v439_data));
              tensorforge::intel_esimd::simd<double, 16> v444_data(glb_m1_run2.template select<16, 1>(16));
              double v445_data = s0_w0[5];
              tensorforge::intel_esimd::simd<double, 16> v447_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v447_data + (v444_data * v445_data));
              double v450_data = s0_w0[21];
              tensorforge::intel_esimd::simd<double, 16> v452_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v452_data + (v444_data * v450_data));
              double v455_data = s0_w0[37];
              tensorforge::intel_esimd::simd<double, 16> v457_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v457_data + (v444_data * v455_data));
              double v460_data = s0_w0[53];
              tensorforge::intel_esimd::simd<double, 16> v462_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v462_data + (v444_data * v460_data));
              double v465_data = s0_w0[69];
              tensorforge::intel_esimd::simd<double, 16> v467_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v467_data + (v444_data * v465_data));
              double v470_data = s0_w0[85];
              tensorforge::intel_esimd::simd<double, 16> v472_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v472_data + (v444_data * v470_data));
              double v475_data = s0_w0[101];
              tensorforge::intel_esimd::simd<double, 16> v477_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v477_data + (v444_data * v475_data));
              double v480_data = s0_w0[117];
              tensorforge::intel_esimd::simd<double, 16> v482_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v482_data + (v444_data * v480_data));
              double v485_data = s0_w0[133];
              tensorforge::intel_esimd::simd<double, 16> v487_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v487_data + (v444_data * v485_data));
              double v490_data = s0_w0[149];
              tensorforge::intel_esimd::simd<double, 16> v492_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v492_data + (v444_data * v490_data));
              double v495_data = s0_w0[165];
              tensorforge::intel_esimd::simd<double, 16> v497_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v497_data + (v444_data * v495_data));
              double v500_data = s0_w0[181];
              tensorforge::intel_esimd::simd<double, 16> v502_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v502_data + (v444_data * v500_data));
              double v505_data = s0_w0[197];
              tensorforge::intel_esimd::simd<double, 16> v507_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v507_data + (v444_data * v505_data));
              double v510_data = s0_w0[213];
              tensorforge::intel_esimd::simd<double, 16> v512_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v512_data + (v444_data * v510_data));
              double v515_data = s0_w0[229];
              tensorforge::intel_esimd::simd<double, 16> v517_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v517_data + (v444_data * v515_data));
              double v520_data = s0_w0[245];
              tensorforge::intel_esimd::simd<double, 16> v522_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v522_data + (v444_data * v520_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run3 = tensorforge::slmLoad<double, 32>(glb_m1 + (96_i32));
              tensorforge::intel_esimd::simd<double, 16> v525_data(glb_m1_run3.template select<16, 1>(0));
              double v526_data = s0_w0[6];
              tensorforge::intel_esimd::simd<double, 16> v528_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v528_data + (v525_data * v526_data));
              double v531_data = s0_w0[22];
              tensorforge::intel_esimd::simd<double, 16> v533_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v533_data + (v525_data * v531_data));
              double v536_data = s0_w0[38];
              tensorforge::intel_esimd::simd<double, 16> v538_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v538_data + (v525_data * v536_data));
              double v541_data = s0_w0[54];
              tensorforge::intel_esimd::simd<double, 16> v543_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v543_data + (v525_data * v541_data));
              double v546_data = s0_w0[70];
              tensorforge::intel_esimd::simd<double, 16> v548_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v548_data + (v525_data * v546_data));
              double v551_data = s0_w0[86];
              tensorforge::intel_esimd::simd<double, 16> v553_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v553_data + (v525_data * v551_data));
              double v556_data = s0_w0[102];
              tensorforge::intel_esimd::simd<double, 16> v558_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v558_data + (v525_data * v556_data));
              double v561_data = s0_w0[118];
              tensorforge::intel_esimd::simd<double, 16> v563_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v563_data + (v525_data * v561_data));
              double v566_data = s0_w0[134];
              tensorforge::intel_esimd::simd<double, 16> v568_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v568_data + (v525_data * v566_data));
              double v571_data = s0_w0[150];
              tensorforge::intel_esimd::simd<double, 16> v573_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v573_data + (v525_data * v571_data));
              double v576_data = s0_w0[166];
              tensorforge::intel_esimd::simd<double, 16> v578_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v578_data + (v525_data * v576_data));
              double v581_data = s0_w0[182];
              tensorforge::intel_esimd::simd<double, 16> v583_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v583_data + (v525_data * v581_data));
              double v586_data = s0_w0[198];
              tensorforge::intel_esimd::simd<double, 16> v588_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v588_data + (v525_data * v586_data));
              double v591_data = s0_w0[214];
              tensorforge::intel_esimd::simd<double, 16> v593_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v593_data + (v525_data * v591_data));
              double v596_data = s0_w0[230];
              tensorforge::intel_esimd::simd<double, 16> v598_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v598_data + (v525_data * v596_data));
              double v601_data = s0_w0[246];
              tensorforge::intel_esimd::simd<double, 16> v603_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v603_data + (v525_data * v601_data));
              tensorforge::intel_esimd::simd<double, 16> v606_data(glb_m1_run3.template select<16, 1>(16));
              double v607_data = s0_w0[7];
              tensorforge::intel_esimd::simd<double, 16> v609_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v609_data + (v606_data * v607_data));
              double v612_data = s0_w0[23];
              tensorforge::intel_esimd::simd<double, 16> v614_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v614_data + (v606_data * v612_data));
              double v617_data = s0_w0[39];
              tensorforge::intel_esimd::simd<double, 16> v619_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v619_data + (v606_data * v617_data));
              double v622_data = s0_w0[55];
              tensorforge::intel_esimd::simd<double, 16> v624_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v624_data + (v606_data * v622_data));
              double v627_data = s0_w0[71];
              tensorforge::intel_esimd::simd<double, 16> v629_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v629_data + (v606_data * v627_data));
              double v632_data = s0_w0[87];
              tensorforge::intel_esimd::simd<double, 16> v634_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v634_data + (v606_data * v632_data));
              double v637_data = s0_w0[103];
              tensorforge::intel_esimd::simd<double, 16> v639_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v639_data + (v606_data * v637_data));
              double v642_data = s0_w0[119];
              tensorforge::intel_esimd::simd<double, 16> v644_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v644_data + (v606_data * v642_data));
              double v647_data = s0_w0[135];
              tensorforge::intel_esimd::simd<double, 16> v649_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v649_data + (v606_data * v647_data));
              double v652_data = s0_w0[151];
              tensorforge::intel_esimd::simd<double, 16> v654_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v654_data + (v606_data * v652_data));
              double v657_data = s0_w0[167];
              tensorforge::intel_esimd::simd<double, 16> v659_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v659_data + (v606_data * v657_data));
              double v662_data = s0_w0[183];
              tensorforge::intel_esimd::simd<double, 16> v664_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v664_data + (v606_data * v662_data));
              double v667_data = s0_w0[199];
              tensorforge::intel_esimd::simd<double, 16> v669_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v669_data + (v606_data * v667_data));
              double v672_data = s0_w0[215];
              tensorforge::intel_esimd::simd<double, 16> v674_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v674_data + (v606_data * v672_data));
              double v677_data = s0_w0[231];
              tensorforge::intel_esimd::simd<double, 16> v679_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v679_data + (v606_data * v677_data));
              double v682_data = s0_w0[247];
              tensorforge::intel_esimd::simd<double, 16> v684_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v684_data + (v606_data * v682_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run4 = tensorforge::slmLoad<double, 32>(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<double, 16> v687_data(glb_m1_run4.template select<16, 1>(0));
              double v688_data = s0_w0[8];
              tensorforge::intel_esimd::simd<double, 16> v690_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v690_data + (v687_data * v688_data));
              double v693_data = s0_w0[24];
              tensorforge::intel_esimd::simd<double, 16> v695_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v695_data + (v687_data * v693_data));
              double v698_data = s0_w0[40];
              tensorforge::intel_esimd::simd<double, 16> v700_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v700_data + (v687_data * v698_data));
              double v703_data = s0_w0[56];
              tensorforge::intel_esimd::simd<double, 16> v705_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v705_data + (v687_data * v703_data));
              double v708_data = s0_w0[72];
              tensorforge::intel_esimd::simd<double, 16> v710_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v710_data + (v687_data * v708_data));
              double v713_data = s0_w0[88];
              tensorforge::intel_esimd::simd<double, 16> v715_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v715_data + (v687_data * v713_data));
              double v718_data = s0_w0[104];
              tensorforge::intel_esimd::simd<double, 16> v720_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v720_data + (v687_data * v718_data));
              double v723_data = s0_w0[120];
              tensorforge::intel_esimd::simd<double, 16> v725_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v725_data + (v687_data * v723_data));
              double v728_data = s0_w0[136];
              tensorforge::intel_esimd::simd<double, 16> v730_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v730_data + (v687_data * v728_data));
              double v733_data = s0_w0[152];
              tensorforge::intel_esimd::simd<double, 16> v735_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v735_data + (v687_data * v733_data));
              double v738_data = s0_w0[168];
              tensorforge::intel_esimd::simd<double, 16> v740_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v740_data + (v687_data * v738_data));
              double v743_data = s0_w0[184];
              tensorforge::intel_esimd::simd<double, 16> v745_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v745_data + (v687_data * v743_data));
              double v748_data = s0_w0[200];
              tensorforge::intel_esimd::simd<double, 16> v750_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v750_data + (v687_data * v748_data));
              double v753_data = s0_w0[216];
              tensorforge::intel_esimd::simd<double, 16> v755_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v755_data + (v687_data * v753_data));
              double v758_data = s0_w0[232];
              tensorforge::intel_esimd::simd<double, 16> v760_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v760_data + (v687_data * v758_data));
              double v763_data = s0_w0[248];
              tensorforge::intel_esimd::simd<double, 16> v765_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v765_data + (v687_data * v763_data));
              tensorforge::intel_esimd::simd<double, 16> v768_data(glb_m1_run4.template select<16, 1>(16));
              double v769_data = s0_w0[9];
              tensorforge::intel_esimd::simd<double, 16> v771_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v771_data + (v768_data * v769_data));
              double v774_data = s0_w0[25];
              tensorforge::intel_esimd::simd<double, 16> v776_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v776_data + (v768_data * v774_data));
              double v779_data = s0_w0[41];
              tensorforge::intel_esimd::simd<double, 16> v781_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v781_data + (v768_data * v779_data));
              double v784_data = s0_w0[57];
              tensorforge::intel_esimd::simd<double, 16> v786_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v786_data + (v768_data * v784_data));
              double v789_data = s0_w0[73];
              tensorforge::intel_esimd::simd<double, 16> v791_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v791_data + (v768_data * v789_data));
              double v794_data = s0_w0[89];
              tensorforge::intel_esimd::simd<double, 16> v796_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v796_data + (v768_data * v794_data));
              double v799_data = s0_w0[105];
              tensorforge::intel_esimd::simd<double, 16> v801_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v801_data + (v768_data * v799_data));
              double v804_data = s0_w0[121];
              tensorforge::intel_esimd::simd<double, 16> v806_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v806_data + (v768_data * v804_data));
              double v809_data = s0_w0[137];
              tensorforge::intel_esimd::simd<double, 16> v811_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v811_data + (v768_data * v809_data));
              double v814_data = s0_w0[153];
              tensorforge::intel_esimd::simd<double, 16> v816_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v816_data + (v768_data * v814_data));
              double v819_data = s0_w0[169];
              tensorforge::intel_esimd::simd<double, 16> v821_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v821_data + (v768_data * v819_data));
              double v824_data = s0_w0[185];
              tensorforge::intel_esimd::simd<double, 16> v826_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v826_data + (v768_data * v824_data));
              double v829_data = s0_w0[201];
              tensorforge::intel_esimd::simd<double, 16> v831_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v831_data + (v768_data * v829_data));
              double v834_data = s0_w0[217];
              tensorforge::intel_esimd::simd<double, 16> v836_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v836_data + (v768_data * v834_data));
              double v839_data = s0_w0[233];
              tensorforge::intel_esimd::simd<double, 16> v841_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v841_data + (v768_data * v839_data));
              double v844_data = s0_w0[249];
              tensorforge::intel_esimd::simd<double, 16> v846_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v846_data + (v768_data * v844_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run5 = tensorforge::slmLoad<double, 32>(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<double, 16> v849_data(glb_m1_run5.template select<16, 1>(0));
              double v850_data = s0_w0[10];
              tensorforge::intel_esimd::simd<double, 16> v852_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v852_data + (v849_data * v850_data));
              double v855_data = s0_w0[26];
              tensorforge::intel_esimd::simd<double, 16> v857_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v857_data + (v849_data * v855_data));
              double v860_data = s0_w0[42];
              tensorforge::intel_esimd::simd<double, 16> v862_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v862_data + (v849_data * v860_data));
              double v865_data = s0_w0[58];
              tensorforge::intel_esimd::simd<double, 16> v867_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v867_data + (v849_data * v865_data));
              double v870_data = s0_w0[74];
              tensorforge::intel_esimd::simd<double, 16> v872_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v872_data + (v849_data * v870_data));
              double v875_data = s0_w0[90];
              tensorforge::intel_esimd::simd<double, 16> v877_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v877_data + (v849_data * v875_data));
              double v880_data = s0_w0[106];
              tensorforge::intel_esimd::simd<double, 16> v882_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v882_data + (v849_data * v880_data));
              double v885_data = s0_w0[122];
              tensorforge::intel_esimd::simd<double, 16> v887_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v887_data + (v849_data * v885_data));
              double v890_data = s0_w0[138];
              tensorforge::intel_esimd::simd<double, 16> v892_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v892_data + (v849_data * v890_data));
              double v895_data = s0_w0[154];
              tensorforge::intel_esimd::simd<double, 16> v897_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v897_data + (v849_data * v895_data));
              double v900_data = s0_w0[170];
              tensorforge::intel_esimd::simd<double, 16> v902_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v902_data + (v849_data * v900_data));
              double v905_data = s0_w0[186];
              tensorforge::intel_esimd::simd<double, 16> v907_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v907_data + (v849_data * v905_data));
              double v910_data = s0_w0[202];
              tensorforge::intel_esimd::simd<double, 16> v912_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v912_data + (v849_data * v910_data));
              double v915_data = s0_w0[218];
              tensorforge::intel_esimd::simd<double, 16> v917_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v917_data + (v849_data * v915_data));
              double v920_data = s0_w0[234];
              tensorforge::intel_esimd::simd<double, 16> v922_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v922_data + (v849_data * v920_data));
              double v925_data = s0_w0[250];
              tensorforge::intel_esimd::simd<double, 16> v927_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v927_data + (v849_data * v925_data));
              tensorforge::intel_esimd::simd<double, 16> v930_data(glb_m1_run5.template select<16, 1>(16));
              double v931_data = s0_w0[11];
              tensorforge::intel_esimd::simd<double, 16> v933_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v933_data + (v930_data * v931_data));
              double v936_data = s0_w0[27];
              tensorforge::intel_esimd::simd<double, 16> v938_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v938_data + (v930_data * v936_data));
              double v941_data = s0_w0[43];
              tensorforge::intel_esimd::simd<double, 16> v943_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v943_data + (v930_data * v941_data));
              double v946_data = s0_w0[59];
              tensorforge::intel_esimd::simd<double, 16> v948_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v948_data + (v930_data * v946_data));
              double v951_data = s0_w0[75];
              tensorforge::intel_esimd::simd<double, 16> v953_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v953_data + (v930_data * v951_data));
              double v956_data = s0_w0[91];
              tensorforge::intel_esimd::simd<double, 16> v958_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v958_data + (v930_data * v956_data));
              double v961_data = s0_w0[107];
              tensorforge::intel_esimd::simd<double, 16> v963_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v963_data + (v930_data * v961_data));
              double v966_data = s0_w0[123];
              tensorforge::intel_esimd::simd<double, 16> v968_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v968_data + (v930_data * v966_data));
              double v971_data = s0_w0[139];
              tensorforge::intel_esimd::simd<double, 16> v973_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v973_data + (v930_data * v971_data));
              double v976_data = s0_w0[155];
              tensorforge::intel_esimd::simd<double, 16> v978_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v978_data + (v930_data * v976_data));
              double v981_data = s0_w0[171];
              tensorforge::intel_esimd::simd<double, 16> v983_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v983_data + (v930_data * v981_data));
              double v986_data = s0_w0[187];
              tensorforge::intel_esimd::simd<double, 16> v988_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v988_data + (v930_data * v986_data));
              double v991_data = s0_w0[203];
              tensorforge::intel_esimd::simd<double, 16> v993_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v993_data + (v930_data * v991_data));
              double v996_data = s0_w0[219];
              tensorforge::intel_esimd::simd<double, 16> v998_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v998_data + (v930_data * v996_data));
              double v1001_data = s0_w0[235];
              tensorforge::intel_esimd::simd<double, 16> v1003_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1003_data + (v930_data * v1001_data));
              double v1006_data = s0_w0[251];
              tensorforge::intel_esimd::simd<double, 16> v1008_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1008_data + (v930_data * v1006_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run6 = tensorforge::slmLoad<double, 32>(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<double, 16> v1011_data(glb_m1_run6.template select<16, 1>(0));
              double v1012_data = s0_w0[12];
              tensorforge::intel_esimd::simd<double, 16> v1014_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v1014_data + (v1011_data * v1012_data));
              double v1017_data = s0_w0[28];
              tensorforge::intel_esimd::simd<double, 16> v1019_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v1019_data + (v1011_data * v1017_data));
              double v1022_data = s0_w0[44];
              tensorforge::intel_esimd::simd<double, 16> v1024_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v1024_data + (v1011_data * v1022_data));
              double v1027_data = s0_w0[60];
              tensorforge::intel_esimd::simd<double, 16> v1029_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v1029_data + (v1011_data * v1027_data));
              double v1032_data = s0_w0[76];
              tensorforge::intel_esimd::simd<double, 16> v1034_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v1034_data + (v1011_data * v1032_data));
              double v1037_data = s0_w0[92];
              tensorforge::intel_esimd::simd<double, 16> v1039_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v1039_data + (v1011_data * v1037_data));
              double v1042_data = s0_w0[108];
              tensorforge::intel_esimd::simd<double, 16> v1044_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v1044_data + (v1011_data * v1042_data));
              double v1047_data = s0_w0[124];
              tensorforge::intel_esimd::simd<double, 16> v1049_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v1049_data + (v1011_data * v1047_data));
              double v1052_data = s0_w0[140];
              tensorforge::intel_esimd::simd<double, 16> v1054_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v1054_data + (v1011_data * v1052_data));
              double v1057_data = s0_w0[156];
              tensorforge::intel_esimd::simd<double, 16> v1059_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v1059_data + (v1011_data * v1057_data));
              double v1062_data = s0_w0[172];
              tensorforge::intel_esimd::simd<double, 16> v1064_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v1064_data + (v1011_data * v1062_data));
              double v1067_data = s0_w0[188];
              tensorforge::intel_esimd::simd<double, 16> v1069_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v1069_data + (v1011_data * v1067_data));
              double v1072_data = s0_w0[204];
              tensorforge::intel_esimd::simd<double, 16> v1074_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v1074_data + (v1011_data * v1072_data));
              double v1077_data = s0_w0[220];
              tensorforge::intel_esimd::simd<double, 16> v1079_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1079_data + (v1011_data * v1077_data));
              double v1082_data = s0_w0[236];
              tensorforge::intel_esimd::simd<double, 16> v1084_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1084_data + (v1011_data * v1082_data));
              double v1087_data = s0_w0[252];
              tensorforge::intel_esimd::simd<double, 16> v1089_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1089_data + (v1011_data * v1087_data));
              tensorforge::intel_esimd::simd<double, 16> v1092_data(glb_m1_run6.template select<16, 1>(16));
              double v1093_data = s0_w0[13];
              tensorforge::intel_esimd::simd<double, 16> v1095_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v1095_data + (v1092_data * v1093_data));
              double v1098_data = s0_w0[29];
              tensorforge::intel_esimd::simd<double, 16> v1100_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v1100_data + (v1092_data * v1098_data));
              double v1103_data = s0_w0[45];
              tensorforge::intel_esimd::simd<double, 16> v1105_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v1105_data + (v1092_data * v1103_data));
              double v1108_data = s0_w0[61];
              tensorforge::intel_esimd::simd<double, 16> v1110_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v1110_data + (v1092_data * v1108_data));
              double v1113_data = s0_w0[77];
              tensorforge::intel_esimd::simd<double, 16> v1115_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v1115_data + (v1092_data * v1113_data));
              double v1118_data = s0_w0[93];
              tensorforge::intel_esimd::simd<double, 16> v1120_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v1120_data + (v1092_data * v1118_data));
              double v1123_data = s0_w0[109];
              tensorforge::intel_esimd::simd<double, 16> v1125_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v1125_data + (v1092_data * v1123_data));
              double v1128_data = s0_w0[125];
              tensorforge::intel_esimd::simd<double, 16> v1130_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v1130_data + (v1092_data * v1128_data));
              double v1133_data = s0_w0[141];
              tensorforge::intel_esimd::simd<double, 16> v1135_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v1135_data + (v1092_data * v1133_data));
              double v1138_data = s0_w0[157];
              tensorforge::intel_esimd::simd<double, 16> v1140_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v1140_data + (v1092_data * v1138_data));
              double v1143_data = s0_w0[173];
              tensorforge::intel_esimd::simd<double, 16> v1145_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v1145_data + (v1092_data * v1143_data));
              double v1148_data = s0_w0[189];
              tensorforge::intel_esimd::simd<double, 16> v1150_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v1150_data + (v1092_data * v1148_data));
              double v1153_data = s0_w0[205];
              tensorforge::intel_esimd::simd<double, 16> v1155_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v1155_data + (v1092_data * v1153_data));
              double v1158_data = s0_w0[221];
              tensorforge::intel_esimd::simd<double, 16> v1160_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1160_data + (v1092_data * v1158_data));
              double v1163_data = s0_w0[237];
              tensorforge::intel_esimd::simd<double, 16> v1165_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1165_data + (v1092_data * v1163_data));
              double v1168_data = s0_w0[253];
              tensorforge::intel_esimd::simd<double, 16> v1170_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1170_data + (v1092_data * v1168_data));
              tensorforge::intel_esimd::simd<double, 32> glb_m1_run7 = tensorforge::slmLoad<double, 32>(glb_m1 + (224_i32));
              tensorforge::intel_esimd::simd<double, 16> v1173_data(glb_m1_run7.template select<16, 1>(0));
              double v1174_data = s0_w0[14];
              tensorforge::intel_esimd::simd<double, 16> v1176_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v1176_data + (v1173_data * v1174_data));
              double v1179_data = s0_w0[30];
              tensorforge::intel_esimd::simd<double, 16> v1181_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v1181_data + (v1173_data * v1179_data));
              double v1184_data = s0_w0[46];
              tensorforge::intel_esimd::simd<double, 16> v1186_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v1186_data + (v1173_data * v1184_data));
              double v1189_data = s0_w0[62];
              tensorforge::intel_esimd::simd<double, 16> v1191_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v1191_data + (v1173_data * v1189_data));
              double v1194_data = s0_w0[78];
              tensorforge::intel_esimd::simd<double, 16> v1196_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v1196_data + (v1173_data * v1194_data));
              double v1199_data = s0_w0[94];
              tensorforge::intel_esimd::simd<double, 16> v1201_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v1201_data + (v1173_data * v1199_data));
              double v1204_data = s0_w0[110];
              tensorforge::intel_esimd::simd<double, 16> v1206_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v1206_data + (v1173_data * v1204_data));
              double v1209_data = s0_w0[126];
              tensorforge::intel_esimd::simd<double, 16> v1211_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v1211_data + (v1173_data * v1209_data));
              double v1214_data = s0_w0[142];
              tensorforge::intel_esimd::simd<double, 16> v1216_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v1216_data + (v1173_data * v1214_data));
              double v1219_data = s0_w0[158];
              tensorforge::intel_esimd::simd<double, 16> v1221_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v1221_data + (v1173_data * v1219_data));
              double v1224_data = s0_w0[174];
              tensorforge::intel_esimd::simd<double, 16> v1226_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v1226_data + (v1173_data * v1224_data));
              double v1229_data = s0_w0[190];
              tensorforge::intel_esimd::simd<double, 16> v1231_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v1231_data + (v1173_data * v1229_data));
              double v1234_data = s0_w0[206];
              tensorforge::intel_esimd::simd<double, 16> v1236_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v1236_data + (v1173_data * v1234_data));
              double v1239_data = s0_w0[222];
              tensorforge::intel_esimd::simd<double, 16> v1241_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1241_data + (v1173_data * v1239_data));
              double v1244_data = s0_w0[238];
              tensorforge::intel_esimd::simd<double, 16> v1246_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1246_data + (v1173_data * v1244_data));
              double v1249_data = s0_w0[254];
              tensorforge::intel_esimd::simd<double, 16> v1251_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1251_data + (v1173_data * v1249_data));
              tensorforge::intel_esimd::simd<double, 16> v1254_data(glb_m1_run7.template select<16, 1>(16));
              double v1255_data = s0_w0[15];
              tensorforge::intel_esimd::simd<double, 16> v1257_data(ir0.template select<16, 1>(0));
              ir0.template select<16, 1>(0) = (v1257_data + (v1254_data * v1255_data));
              double v1260_data = s0_w0[31];
              tensorforge::intel_esimd::simd<double, 16> v1262_data(ir0.template select<16, 1>(16));
              ir0.template select<16, 1>(16) = (v1262_data + (v1254_data * v1260_data));
              double v1265_data = s0_w0[47];
              tensorforge::intel_esimd::simd<double, 16> v1267_data(ir0.template select<16, 1>(32));
              ir0.template select<16, 1>(32) = (v1267_data + (v1254_data * v1265_data));
              double v1270_data = s0_w0[63];
              tensorforge::intel_esimd::simd<double, 16> v1272_data(ir0.template select<16, 1>(48));
              ir0.template select<16, 1>(48) = (v1272_data + (v1254_data * v1270_data));
              double v1275_data = s0_w0[79];
              tensorforge::intel_esimd::simd<double, 16> v1277_data(ir0.template select<16, 1>(64));
              ir0.template select<16, 1>(64) = (v1277_data + (v1254_data * v1275_data));
              double v1280_data = s0_w0[95];
              tensorforge::intel_esimd::simd<double, 16> v1282_data(ir0.template select<16, 1>(80));
              ir0.template select<16, 1>(80) = (v1282_data + (v1254_data * v1280_data));
              double v1285_data = s0_w0[111];
              tensorforge::intel_esimd::simd<double, 16> v1287_data(ir0.template select<16, 1>(96));
              ir0.template select<16, 1>(96) = (v1287_data + (v1254_data * v1285_data));
              double v1290_data = s0_w0[127];
              tensorforge::intel_esimd::simd<double, 16> v1292_data(ir0.template select<16, 1>(112));
              ir0.template select<16, 1>(112) = (v1292_data + (v1254_data * v1290_data));
              double v1295_data = s0_w0[143];
              tensorforge::intel_esimd::simd<double, 16> v1297_data(ir0.template select<16, 1>(128));
              ir0.template select<16, 1>(128) = (v1297_data + (v1254_data * v1295_data));
              double v1300_data = s0_w0[159];
              tensorforge::intel_esimd::simd<double, 16> v1302_data(ir0.template select<16, 1>(144));
              ir0.template select<16, 1>(144) = (v1302_data + (v1254_data * v1300_data));
              double v1305_data = s0_w0[175];
              tensorforge::intel_esimd::simd<double, 16> v1307_data(ir0.template select<16, 1>(160));
              ir0.template select<16, 1>(160) = (v1307_data + (v1254_data * v1305_data));
              double v1310_data = s0_w0[191];
              tensorforge::intel_esimd::simd<double, 16> v1312_data(ir0.template select<16, 1>(176));
              ir0.template select<16, 1>(176) = (v1312_data + (v1254_data * v1310_data));
              double v1315_data = s0_w0[207];
              tensorforge::intel_esimd::simd<double, 16> v1317_data(ir0.template select<16, 1>(192));
              ir0.template select<16, 1>(192) = (v1317_data + (v1254_data * v1315_data));
              double v1320_data = s0_w0[223];
              tensorforge::intel_esimd::simd<double, 16> v1322_data(ir0.template select<16, 1>(208));
              ir0.template select<16, 1>(208) = (v1322_data + (v1254_data * v1320_data));
              double v1325_data = s0_w0[239];
              tensorforge::intel_esimd::simd<double, 16> v1327_data(ir0.template select<16, 1>(224));
              ir0.template select<16, 1>(224) = (v1327_data + (v1254_data * v1325_data));
              double v1330_data = s0_w0[255];
              tensorforge::intel_esimd::simd<double, 16> v1332_data(ir0.template select<16, 1>(240));
              ir0.template select<16, 1>(240) = (v1332_data + (v1254_data * v1330_data));
              // r0 = ir0
              #pragma unroll
              for (int32_t v1334_n0 = 0; v1334_n0 < 1; ++v1334_n0) {
                int32_t v1336_a = v1334_n0 * 16;
                #pragma unroll
                for (int32_t v1335_n1 = 0; v1335_n1 < 16; ++v1335_n1) {
                  int32_t v1338_a = v1336_a + (v1335_n1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v1339_data(ir0.template select<16, 1>(v1338_a));
                  r0.template select<16, 1>(v1338_a) = v1339_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v1340_i0 = 0; v1340_i0 < 1; ++v1340_i0) {
                int32_t v1342_a = v1340_i0 * 16;
                #pragma unroll
                for (int32_t v1341_i1 = 0; v1341_i1 < 16; ++v1341_i1) {
                  int32_t v1344_a = v1342_a + (v1341_i1 * 16);
                  tensorforge::intel_esimd::simd<double, 16> v1345_data(r0.template select<16, 1>(v1344_a));
                  v1345_data.copy_to(glb_m0 + (v1344_a));
                }
              }
            }
          }
        }
      }
    });
  });
}

