// === base name ===
kernel_924fd3d329

// === header ===
void launcher_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_924fd3d329(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_924fd3d329(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×4(8×4) {0..8}×{0..4} strided
        // m2 8×4(8×4) {0..8}×{0..4} strided
        // m3 8×8(8×8) {0..8}×{0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
        // C = abs(TMP)
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 32 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
              float r0[128]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v7_i1 = 0; v7_i1 < 8; ++v7_i1) {
                tensorforge::intel_esimd::simd<float, 8> v12_data;
                v12_data.copy_from(glb_m0 + ((v7_i1 * 8)));
                v12_data.copy_to(r0 + ((v7_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v16_ld;
              v16_ld.copy_from(glb_m1 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v16_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m0););
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[64]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 16> v18_data;
              v18_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v19_data;
              v19_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v20_data;
              v20_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v21_data;
              v21_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v22_data;
              v22_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v24_data;
              v24_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v26_acc{};
              tensorforge::intel_esimd::simd<float, 16> v27_lin;
              v27_lin.copy_from(s0 + (0 + item.get_local_id(0) * 1));
              float v28_bc = v27_lin[0];
              v26_acc += (v28_bc * v18_data);
              float v30_bc = v27_lin[1];
              v26_acc += (v30_bc * v19_data);
              float v32_bc = v27_lin[2];
              v26_acc += (v32_bc * v20_data);
              float v34_bc = v27_lin[3];
              v26_acc += (v34_bc * v21_data);
              float v36_bc = v27_lin[4];
              v26_acc += (v36_bc * v22_data);
              float v38_bc = v27_lin[5];
              v26_acc += (v38_bc * v23_data);
              float v40_bc = v27_lin[6];
              v26_acc += (v40_bc * v24_data);
              float v42_bc = v27_lin[7];
              v26_acc += (v42_bc * v25_data);
              v26_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v44_acc{};
              v44_acc += (v28_bc * v18_data);
              v44_acc += (v30_bc * v19_data);
              v44_acc += (v32_bc * v20_data);
              v44_acc += (v34_bc * v21_data);
              v44_acc += (v36_bc * v22_data);
              v44_acc += (v38_bc * v23_data);
              v44_acc += (v40_bc * v24_data);
              v44_acc += (v42_bc * v25_data);
              v44_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v62_acc{};
              v62_acc += (v28_bc * v18_data);
              v62_acc += (v30_bc * v19_data);
              v62_acc += (v32_bc * v20_data);
              v62_acc += (v34_bc * v21_data);
              v62_acc += (v36_bc * v22_data);
              v62_acc += (v38_bc * v23_data);
              v62_acc += (v40_bc * v24_data);
              v62_acc += (v42_bc * v25_data);
              v62_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v80_acc{};
              v80_acc += (v28_bc * v18_data);
              v80_acc += (v30_bc * v19_data);
              v80_acc += (v32_bc * v20_data);
              v80_acc += (v34_bc * v21_data);
              v80_acc += (v36_bc * v22_data);
              v80_acc += (v38_bc * v23_data);
              v80_acc += (v40_bc * v24_data);
              v80_acc += (v42_bc * v25_data);
              v80_acc.copy_to(r1 + (48));
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v99_i1 = 0; v99_i1 < 4; ++v99_i1) {
                tensorforge::intel_esimd::simd<float, 8> v102_data;
                v102_data.copy_from(r1 + ((v99_i1 * 16)));
                v102_data.copy_to(s1 + ((v99_i1 * 8)));
              }
              float* __restrict__ s2 = &localShrMem0[0];
              // s2 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v108_ld;
              v108_ld.copy_from(glb_m2 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v108_ld.copy_to(s2 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              // wait(s2 = load{g>s}(glb_m2[0, 1]));
              float r2[64]{};
              // r2 = +(r0 * s2) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              float ir2[64]{};
              tensorforge::intel_esimd::simd<float, 16> v119_acc{};
              tensorforge::intel_esimd::simd<float, 16> v120_lin;
              v120_lin.copy_from(s2 + (0 + item.get_local_id(0) * 1));
              float v121_bc = v120_lin[0];
              v119_acc += (v121_bc * v18_data);
              float v123_bc = v120_lin[1];
              v119_acc += (v123_bc * v19_data);
              float v125_bc = v120_lin[2];
              v119_acc += (v125_bc * v20_data);
              float v127_bc = v120_lin[3];
              v119_acc += (v127_bc * v21_data);
              float v129_bc = v120_lin[4];
              v119_acc += (v129_bc * v22_data);
              float v131_bc = v120_lin[5];
              v119_acc += (v131_bc * v23_data);
              float v133_bc = v120_lin[6];
              v119_acc += (v133_bc * v24_data);
              float v135_bc = v120_lin[7];
              v119_acc += (v135_bc * v25_data);
              v119_acc.copy_to(ir2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v137_acc{};
              v137_acc += (v121_bc * v18_data);
              v137_acc += (v123_bc * v19_data);
              v137_acc += (v125_bc * v20_data);
              v137_acc += (v127_bc * v21_data);
              v137_acc += (v129_bc * v22_data);
              v137_acc += (v131_bc * v23_data);
              v137_acc += (v133_bc * v24_data);
              v137_acc += (v135_bc * v25_data);
              v137_acc.copy_to(ir2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v155_acc{};
              v155_acc += (v121_bc * v18_data);
              v155_acc += (v123_bc * v19_data);
              v155_acc += (v125_bc * v20_data);
              v155_acc += (v127_bc * v21_data);
              v155_acc += (v129_bc * v22_data);
              v155_acc += (v131_bc * v23_data);
              v155_acc += (v133_bc * v24_data);
              v155_acc += (v135_bc * v25_data);
              v155_acc.copy_to(ir2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v173_acc{};
              v173_acc += (v121_bc * v18_data);
              v173_acc += (v123_bc * v19_data);
              v173_acc += (v125_bc * v20_data);
              v173_acc += (v127_bc * v21_data);
              v173_acc += (v129_bc * v22_data);
              v173_acc += (v131_bc * v23_data);
              v173_acc += (v133_bc * v24_data);
              v173_acc += (v135_bc * v25_data);
              v173_acc.copy_to(ir2 + (48));
              #pragma unroll
              for (int32_t v191_n1 = 0; v191_n1 < 4; ++v191_n1) {
                int32_t v192_a = v191_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v194_data;
                v194_data.copy_from(ir2 + (v192_a));
                v194_data.copy_to(r2 + (v192_a));
              }
              // s1 = store{r>s}(localShrMem0, r2);
              #pragma unroll
              for (int32_t v197_i1 = 0; v197_i1 < 4; ++v197_i1) {
                tensorforge::intel_esimd::simd<float, 8> v200_data;
                v200_data.copy_from(r2 + ((v197_i1 * 16)));
                v200_data.copy_to(s1 + (((v197_i1 + 4) * 8)));
              }
              // glb_m3 = abs(s1)
              #pragma unroll
              for (int32_t v206_k1 = 0; v206_k1 < 8; ++v206_k1) {
                int32_t v209_a = v206_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v211_data;
                v211_data.copy_from(s1 + (v209_a));
                (tensorforge::intel_esimd::abs(v211_data)).copy_to(glb_m3 + (v209_a));
              }
            }
          }
        }
      });
    }
  });
}

