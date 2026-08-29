// === base name ===
kernel_ead773dd51

// === header ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ead773dd51(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ead773dd51(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 32×16(32×16) {0..32}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(32×16) {0..32}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[144 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[128];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 512 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[128]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir0[128]{};
              tensorforge::intel_esimd::simd<float, 16> v10_data;
              v10_data.copy_from(glb_m1 + (4_i32));
              tensorforge::intel_esimd::simd<float, 16> v15_data;
              v15_data.copy_from(glb_m1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v20_data;
              v20_data.copy_from(glb_m1 + (68_i32));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(glb_m1 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(glb_m1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(glb_m1 + (164_i32));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(glb_m1 + (196_i32));
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(glb_m1 + (228_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(glb_m1 + (260_i32));
              tensorforge::intel_esimd::simd<float, 16> v55_data;
              v55_data.copy_from(glb_m1 + (292_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_data;
              v60_data.copy_from(glb_m1 + (324_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data;
              v65_data.copy_from(glb_m1 + (356_i32));
              tensorforge::intel_esimd::simd<float, 16> v70_data;
              v70_data.copy_from(glb_m1 + (388_i32));
              tensorforge::intel_esimd::simd<float, 16> v75_data;
              v75_data.copy_from(glb_m1 + (420_i32));
              tensorforge::intel_esimd::simd<float, 16> v80_data;
              v80_data.copy_from(glb_m1 + (452_i32));
              tensorforge::intel_esimd::simd<float, 16> v85_data;
              v85_data.copy_from(glb_m1 + (484_i32));
              tensorforge::intel_esimd::simd<float, 16> v86_acc{};
              tensorforge::intel_esimd::simd<float, 16> v93_data;
              v93_data.copy_from(s0 + ((0_i32 ^ ((0_i32 >> 5) & 31))));
              v86_acc += ((v93_data[0]) * v10_data);
              v86_acc += ((v93_data[1]) * v15_data);
              v86_acc += ((v93_data[2]) * v20_data);
              v86_acc += ((v93_data[3]) * v25_data);
              v86_acc += ((v93_data[4]) * v30_data);
              v86_acc += ((v93_data[5]) * v35_data);
              v86_acc += ((v93_data[6]) * v40_data);
              v86_acc += ((v93_data[7]) * v45_data);
              v86_acc += ((v93_data[8]) * v50_data);
              v86_acc += ((v93_data[9]) * v55_data);
              v86_acc += ((v93_data[10]) * v60_data);
              v86_acc += ((v93_data[11]) * v65_data);
              v86_acc += ((v93_data[12]) * v70_data);
              v86_acc += ((v93_data[13]) * v75_data);
              v86_acc += ((v93_data[14]) * v80_data);
              v86_acc += ((v93_data[15]) * v85_data);
              v86_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v126_acc{};
              tensorforge::intel_esimd::simd<float, 16> v133_data;
              v133_data.copy_from(s0 + ((16_i32 ^ ((16_i32 >> 5) & 31))));
              v126_acc += ((v133_data[0]) * v10_data);
              v126_acc += ((v133_data[1]) * v15_data);
              v126_acc += ((v133_data[2]) * v20_data);
              v126_acc += ((v133_data[3]) * v25_data);
              v126_acc += ((v133_data[4]) * v30_data);
              v126_acc += ((v133_data[5]) * v35_data);
              v126_acc += ((v133_data[6]) * v40_data);
              v126_acc += ((v133_data[7]) * v45_data);
              v126_acc += ((v133_data[8]) * v50_data);
              v126_acc += ((v133_data[9]) * v55_data);
              v126_acc += ((v133_data[10]) * v60_data);
              v126_acc += ((v133_data[11]) * v65_data);
              v126_acc += ((v133_data[12]) * v70_data);
              v126_acc += ((v133_data[13]) * v75_data);
              v126_acc += ((v133_data[14]) * v80_data);
              v126_acc += ((v133_data[15]) * v85_data);
              v126_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v166_acc{};
              tensorforge::intel_esimd::simd<float, 16> v173_data;
              v173_data.copy_from(s0 + ((32_i32 ^ ((32_i32 >> 5) & 31))));
              v166_acc += ((v173_data[0]) * v10_data);
              v166_acc += ((v173_data[1]) * v15_data);
              v166_acc += ((v173_data[2]) * v20_data);
              v166_acc += ((v173_data[3]) * v25_data);
              v166_acc += ((v173_data[4]) * v30_data);
              v166_acc += ((v173_data[5]) * v35_data);
              v166_acc += ((v173_data[6]) * v40_data);
              v166_acc += ((v173_data[7]) * v45_data);
              v166_acc += ((v173_data[8]) * v50_data);
              v166_acc += ((v173_data[9]) * v55_data);
              v166_acc += ((v173_data[10]) * v60_data);
              v166_acc += ((v173_data[11]) * v65_data);
              v166_acc += ((v173_data[12]) * v70_data);
              v166_acc += ((v173_data[13]) * v75_data);
              v166_acc += ((v173_data[14]) * v80_data);
              v166_acc += ((v173_data[15]) * v85_data);
              v166_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v206_acc{};
              tensorforge::intel_esimd::simd<float, 16> v213_data;
              v213_data.copy_from(s0 + ((48_i32 ^ ((48_i32 >> 5) & 31))));
              v206_acc += ((v213_data[0]) * v10_data);
              v206_acc += ((v213_data[1]) * v15_data);
              v206_acc += ((v213_data[2]) * v20_data);
              v206_acc += ((v213_data[3]) * v25_data);
              v206_acc += ((v213_data[4]) * v30_data);
              v206_acc += ((v213_data[5]) * v35_data);
              v206_acc += ((v213_data[6]) * v40_data);
              v206_acc += ((v213_data[7]) * v45_data);
              v206_acc += ((v213_data[8]) * v50_data);
              v206_acc += ((v213_data[9]) * v55_data);
              v206_acc += ((v213_data[10]) * v60_data);
              v206_acc += ((v213_data[11]) * v65_data);
              v206_acc += ((v213_data[12]) * v70_data);
              v206_acc += ((v213_data[13]) * v75_data);
              v206_acc += ((v213_data[14]) * v80_data);
              v206_acc += ((v213_data[15]) * v85_data);
              v206_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v246_acc{};
              tensorforge::intel_esimd::simd<float, 16> v253_data;
              v253_data.copy_from(s0 + ((64_i32 ^ ((64_i32 >> 5) & 31))));
              v246_acc += ((v253_data[0]) * v10_data);
              v246_acc += ((v253_data[1]) * v15_data);
              v246_acc += ((v253_data[2]) * v20_data);
              v246_acc += ((v253_data[3]) * v25_data);
              v246_acc += ((v253_data[4]) * v30_data);
              v246_acc += ((v253_data[5]) * v35_data);
              v246_acc += ((v253_data[6]) * v40_data);
              v246_acc += ((v253_data[7]) * v45_data);
              v246_acc += ((v253_data[8]) * v50_data);
              v246_acc += ((v253_data[9]) * v55_data);
              v246_acc += ((v253_data[10]) * v60_data);
              v246_acc += ((v253_data[11]) * v65_data);
              v246_acc += ((v253_data[12]) * v70_data);
              v246_acc += ((v253_data[13]) * v75_data);
              v246_acc += ((v253_data[14]) * v80_data);
              v246_acc += ((v253_data[15]) * v85_data);
              v246_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v286_acc{};
              tensorforge::intel_esimd::simd<float, 16> v293_data;
              v293_data.copy_from(s0 + ((80_i32 ^ ((80_i32 >> 5) & 31))));
              v286_acc += ((v293_data[0]) * v10_data);
              v286_acc += ((v293_data[1]) * v15_data);
              v286_acc += ((v293_data[2]) * v20_data);
              v286_acc += ((v293_data[3]) * v25_data);
              v286_acc += ((v293_data[4]) * v30_data);
              v286_acc += ((v293_data[5]) * v35_data);
              v286_acc += ((v293_data[6]) * v40_data);
              v286_acc += ((v293_data[7]) * v45_data);
              v286_acc += ((v293_data[8]) * v50_data);
              v286_acc += ((v293_data[9]) * v55_data);
              v286_acc += ((v293_data[10]) * v60_data);
              v286_acc += ((v293_data[11]) * v65_data);
              v286_acc += ((v293_data[12]) * v70_data);
              v286_acc += ((v293_data[13]) * v75_data);
              v286_acc += ((v293_data[14]) * v80_data);
              v286_acc += ((v293_data[15]) * v85_data);
              v286_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v326_acc{};
              tensorforge::intel_esimd::simd<float, 16> v333_data;
              v333_data.copy_from(s0 + ((96_i32 ^ ((96_i32 >> 5) & 31))));
              v326_acc += ((v333_data[0]) * v10_data);
              v326_acc += ((v333_data[1]) * v15_data);
              v326_acc += ((v333_data[2]) * v20_data);
              v326_acc += ((v333_data[3]) * v25_data);
              v326_acc += ((v333_data[4]) * v30_data);
              v326_acc += ((v333_data[5]) * v35_data);
              v326_acc += ((v333_data[6]) * v40_data);
              v326_acc += ((v333_data[7]) * v45_data);
              v326_acc += ((v333_data[8]) * v50_data);
              v326_acc += ((v333_data[9]) * v55_data);
              v326_acc += ((v333_data[10]) * v60_data);
              v326_acc += ((v333_data[11]) * v65_data);
              v326_acc += ((v333_data[12]) * v70_data);
              v326_acc += ((v333_data[13]) * v75_data);
              v326_acc += ((v333_data[14]) * v80_data);
              v326_acc += ((v333_data[15]) * v85_data);
              v326_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v366_acc{};
              tensorforge::intel_esimd::simd<float, 16> v373_data;
              v373_data.copy_from(s0 + ((112_i32 ^ ((112_i32 >> 5) & 31))));
              v366_acc += ((v373_data[0]) * v10_data);
              v366_acc += ((v373_data[1]) * v15_data);
              v366_acc += ((v373_data[2]) * v20_data);
              v366_acc += ((v373_data[3]) * v25_data);
              v366_acc += ((v373_data[4]) * v30_data);
              v366_acc += ((v373_data[5]) * v35_data);
              v366_acc += ((v373_data[6]) * v40_data);
              v366_acc += ((v373_data[7]) * v45_data);
              v366_acc += ((v373_data[8]) * v50_data);
              v366_acc += ((v373_data[9]) * v55_data);
              v366_acc += ((v373_data[10]) * v60_data);
              v366_acc += ((v373_data[11]) * v65_data);
              v366_acc += ((v373_data[12]) * v70_data);
              v366_acc += ((v373_data[13]) * v75_data);
              v366_acc += ((v373_data[14]) * v80_data);
              v366_acc += ((v373_data[15]) * v85_data);
              v366_acc.copy_to(ir0 + (112));
              #pragma unroll
              for (int32_t v406_n1 = 0; v406_n1 < 8; ++v406_n1) {
                int32_t v407_a = v406_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v409_data;
                v409_data.copy_from(ir0 + (v407_a));
                v409_data.copy_to(r0 + (v407_a));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v412_i1 = 0; v412_i1 < 8; ++v412_i1) {
                tensorforge::intel_esimd::simd<float, 12> v415_data;
                v415_data.copy_from(r0 + ((v412_i1 * 16)));
                v415_data.copy_to(glb_m0 + ((v412_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

