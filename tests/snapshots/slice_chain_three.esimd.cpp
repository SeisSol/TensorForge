// === base name ===
kernel_08703cce1d

// === header ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08703cce1d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08703cce1d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1536, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×32(12×6) {0..12}×{0..6} strided
        // m1 32×32(6×6) {0..6}×{0..6} strided
        // m2 32×32(12×6) {0..12}×{0..6} strided
        // m3 32×32(12×12) {0..12}×{0..12} strided
        // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
        // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[96 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[80];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              float r0[96]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v7_i1 = 0; v7_i1 < 6; ++v7_i1) {
                tensorforge::intel_esimd::simd<float, 12> v12_data;
                v12_data.copy_from(glb_m0 + ((v7_i1 * 12)));
                v12_data.copy_to(r0 + ((v7_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v16_ld;
              v16_ld.copy_from(glb_m1 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v16_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              if (item.get_local_id(0) < 4) {
                tensorforge::intel_esimd::simd<float, 16> v17_ld;
                v17_ld.copy_from(glb_m1 + (0 + 0 + 1 * item.get_local_id(0) + 32));
                v17_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 32));
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r2[192]{};
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 12; ++v19_i1) {
                tensorforge::intel_esimd::simd<float, 12> v24_data;
                v24_data.copy_from(glb_m3 + ((v19_i1 * 12)));
                v24_data.copy_to(r2 + ((v19_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[96]{};
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v34_acc{};
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(s0 + (0_i32));
              v34_acc += ((v38_data[0]) * v28_data);
              v34_acc += ((v38_data[1]) * v29_data);
              v34_acc += ((v38_data[2]) * v30_data);
              v34_acc += ((v38_data[3]) * v31_data);
              v34_acc += ((v38_data[4]) * v32_data);
              v34_acc += ((v38_data[5]) * v33_data);
              v34_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v51_acc{};
              tensorforge::intel_esimd::simd<float, 16> v55_data;
              v55_data.copy_from(s0 + (6_i32));
              v51_acc += ((v55_data[0]) * v28_data);
              v51_acc += ((v55_data[1]) * v29_data);
              v51_acc += ((v55_data[2]) * v30_data);
              v51_acc += ((v55_data[3]) * v31_data);
              v51_acc += ((v55_data[4]) * v32_data);
              v51_acc += ((v55_data[5]) * v33_data);
              v51_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v68_acc{};
              tensorforge::intel_esimd::simd<float, 16> v72_data;
              v72_data.copy_from(s0 + (12_i32));
              v68_acc += ((v72_data[0]) * v28_data);
              v68_acc += ((v72_data[1]) * v29_data);
              v68_acc += ((v72_data[2]) * v30_data);
              v68_acc += ((v72_data[3]) * v31_data);
              v68_acc += ((v72_data[4]) * v32_data);
              v68_acc += ((v72_data[5]) * v33_data);
              v68_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v85_acc{};
              tensorforge::intel_esimd::simd<float, 16> v89_data;
              v89_data.copy_from(s0 + (18_i32));
              v85_acc += ((v89_data[0]) * v28_data);
              v85_acc += ((v89_data[1]) * v29_data);
              v85_acc += ((v89_data[2]) * v30_data);
              v85_acc += ((v89_data[3]) * v31_data);
              v85_acc += ((v89_data[4]) * v32_data);
              v85_acc += ((v89_data[5]) * v33_data);
              v85_acc.copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v102_acc{};
              tensorforge::intel_esimd::simd<float, 16> v106_data;
              v106_data.copy_from(s0 + (24_i32));
              v102_acc += ((v106_data[0]) * v28_data);
              v102_acc += ((v106_data[1]) * v29_data);
              v102_acc += ((v106_data[2]) * v30_data);
              v102_acc += ((v106_data[3]) * v31_data);
              v102_acc += ((v106_data[4]) * v32_data);
              v102_acc += ((v106_data[5]) * v33_data);
              v102_acc.copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v119_acc{};
              tensorforge::intel_esimd::simd<float, 16> v123_data;
              v123_data.copy_from(s0 + (30_i32));
              v119_acc += ((v123_data[0]) * v28_data);
              v119_acc += ((v123_data[1]) * v29_data);
              v119_acc += ((v123_data[2]) * v30_data);
              v119_acc += ((v123_data[3]) * v31_data);
              v119_acc += ((v123_data[4]) * v32_data);
              v119_acc += ((v123_data[5]) * v33_data);
              v119_acc.copy_to(r1 + (80));
              // wait(r2 = load{g>r}(glb_m3););
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v137_i1 = 0; v137_i1 < 6; ++v137_i1) {
                tensorforge::intel_esimd::simd<float, 12> v140_data;
                v140_data.copy_from(r1 + ((v137_i1 * 16)));
                v140_data.copy_to(s1 + ((v137_i1 * 12)));
              }
              float r3[96]{};
              // r3 = +(r2 * s1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir3[96]{};
              tensorforge::intel_esimd::simd<float, 16> v147_data;
              v147_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v148_data;
              v148_data.copy_from(r2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v149_data;
              v149_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v150_data;
              v150_data.copy_from(r2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v151_data;
              v151_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v152_data;
              v152_data.copy_from(r2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v153_data;
              v153_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v154_data;
              v154_data.copy_from(r2 + (112));
              tensorforge::intel_esimd::simd<float, 16> v155_data;
              v155_data.copy_from(r2 + (128));
              tensorforge::intel_esimd::simd<float, 16> v156_data;
              v156_data.copy_from(r2 + (144));
              tensorforge::intel_esimd::simd<float, 16> v157_data;
              v157_data.copy_from(r2 + (160));
              tensorforge::intel_esimd::simd<float, 16> v158_data;
              v158_data.copy_from(r2 + (176));
              tensorforge::intel_esimd::simd<float, 16> v159_acc{};
              tensorforge::intel_esimd::simd<float, 16> v163_data;
              v163_data.copy_from(s1 + (0_i32));
              v159_acc += ((v163_data[0]) * v147_data);
              v159_acc += ((v163_data[1]) * v148_data);
              v159_acc += ((v163_data[2]) * v149_data);
              v159_acc += ((v163_data[3]) * v150_data);
              v159_acc += ((v163_data[4]) * v151_data);
              v159_acc += ((v163_data[5]) * v152_data);
              v159_acc += ((v163_data[6]) * v153_data);
              v159_acc += ((v163_data[7]) * v154_data);
              v159_acc += ((v163_data[8]) * v155_data);
              v159_acc += ((v163_data[9]) * v156_data);
              v159_acc += ((v163_data[10]) * v157_data);
              v159_acc += ((v163_data[11]) * v158_data);
              v159_acc.copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v188_acc{};
              tensorforge::intel_esimd::simd<float, 16> v192_data;
              v192_data.copy_from(s1 + (12_i32));
              v188_acc += ((v192_data[0]) * v147_data);
              v188_acc += ((v192_data[1]) * v148_data);
              v188_acc += ((v192_data[2]) * v149_data);
              v188_acc += ((v192_data[3]) * v150_data);
              v188_acc += ((v192_data[4]) * v151_data);
              v188_acc += ((v192_data[5]) * v152_data);
              v188_acc += ((v192_data[6]) * v153_data);
              v188_acc += ((v192_data[7]) * v154_data);
              v188_acc += ((v192_data[8]) * v155_data);
              v188_acc += ((v192_data[9]) * v156_data);
              v188_acc += ((v192_data[10]) * v157_data);
              v188_acc += ((v192_data[11]) * v158_data);
              v188_acc.copy_to(ir3 + (16));
              tensorforge::intel_esimd::simd<float, 16> v217_acc{};
              tensorforge::intel_esimd::simd<float, 16> v221_data;
              v221_data.copy_from(s1 + (24_i32));
              v217_acc += ((v221_data[0]) * v147_data);
              v217_acc += ((v221_data[1]) * v148_data);
              v217_acc += ((v221_data[2]) * v149_data);
              v217_acc += ((v221_data[3]) * v150_data);
              v217_acc += ((v221_data[4]) * v151_data);
              v217_acc += ((v221_data[5]) * v152_data);
              v217_acc += ((v221_data[6]) * v153_data);
              v217_acc += ((v221_data[7]) * v154_data);
              v217_acc += ((v221_data[8]) * v155_data);
              v217_acc += ((v221_data[9]) * v156_data);
              v217_acc += ((v221_data[10]) * v157_data);
              v217_acc += ((v221_data[11]) * v158_data);
              v217_acc.copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v246_acc{};
              tensorforge::intel_esimd::simd<float, 16> v250_data;
              v250_data.copy_from(s1 + (36_i32));
              v246_acc += ((v250_data[0]) * v147_data);
              v246_acc += ((v250_data[1]) * v148_data);
              v246_acc += ((v250_data[2]) * v149_data);
              v246_acc += ((v250_data[3]) * v150_data);
              v246_acc += ((v250_data[4]) * v151_data);
              v246_acc += ((v250_data[5]) * v152_data);
              v246_acc += ((v250_data[6]) * v153_data);
              v246_acc += ((v250_data[7]) * v154_data);
              v246_acc += ((v250_data[8]) * v155_data);
              v246_acc += ((v250_data[9]) * v156_data);
              v246_acc += ((v250_data[10]) * v157_data);
              v246_acc += ((v250_data[11]) * v158_data);
              v246_acc.copy_to(ir3 + (48));
              tensorforge::intel_esimd::simd<float, 16> v275_acc{};
              tensorforge::intel_esimd::simd<float, 16> v279_data;
              v279_data.copy_from(s1 + (48_i32));
              v275_acc += ((v279_data[0]) * v147_data);
              v275_acc += ((v279_data[1]) * v148_data);
              v275_acc += ((v279_data[2]) * v149_data);
              v275_acc += ((v279_data[3]) * v150_data);
              v275_acc += ((v279_data[4]) * v151_data);
              v275_acc += ((v279_data[5]) * v152_data);
              v275_acc += ((v279_data[6]) * v153_data);
              v275_acc += ((v279_data[7]) * v154_data);
              v275_acc += ((v279_data[8]) * v155_data);
              v275_acc += ((v279_data[9]) * v156_data);
              v275_acc += ((v279_data[10]) * v157_data);
              v275_acc += ((v279_data[11]) * v158_data);
              v275_acc.copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v304_acc{};
              tensorforge::intel_esimd::simd<float, 16> v308_data;
              v308_data.copy_from(s1 + (60_i32));
              v304_acc += ((v308_data[0]) * v147_data);
              v304_acc += ((v308_data[1]) * v148_data);
              v304_acc += ((v308_data[2]) * v149_data);
              v304_acc += ((v308_data[3]) * v150_data);
              v304_acc += ((v308_data[4]) * v151_data);
              v304_acc += ((v308_data[5]) * v152_data);
              v304_acc += ((v308_data[6]) * v153_data);
              v304_acc += ((v308_data[7]) * v154_data);
              v304_acc += ((v308_data[8]) * v155_data);
              v304_acc += ((v308_data[9]) * v156_data);
              v304_acc += ((v308_data[10]) * v157_data);
              v304_acc += ((v308_data[11]) * v158_data);
              v304_acc.copy_to(ir3 + (80));
              #pragma unroll
              for (int32_t v333_n1 = 0; v333_n1 < 6; ++v333_n1) {
                int32_t v334_a = v333_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v336_data;
                v336_data.copy_from(ir3 + (v334_a));
                v336_data.copy_to(r3 + (v334_a));
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v339_i1 = 0; v339_i1 < 6; ++v339_i1) {
                tensorforge::intel_esimd::simd<float, 12> v342_data;
                v342_data.copy_from(r3 + ((v339_i1 * 16)));
                v342_data.copy_to(glb_m2 + ((v339_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

