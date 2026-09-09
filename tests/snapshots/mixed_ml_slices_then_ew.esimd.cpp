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
        // options: default
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
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(s0 + (0_i32));
              v26_acc += ((v30_data[0]) * v18_data);
              v26_acc += ((v30_data[1]) * v19_data);
              v26_acc += ((v30_data[2]) * v20_data);
              v26_acc += ((v30_data[3]) * v21_data);
              v26_acc += ((v30_data[4]) * v22_data);
              v26_acc += ((v30_data[5]) * v23_data);
              v26_acc += ((v30_data[6]) * v24_data);
              v26_acc += ((v30_data[7]) * v25_data);
              v26_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v47_acc{};
              tensorforge::intel_esimd::simd<float, 16> v51_data;
              v51_data.copy_from(s0 + (8_i32));
              v47_acc += ((v51_data[0]) * v18_data);
              v47_acc += ((v51_data[1]) * v19_data);
              v47_acc += ((v51_data[2]) * v20_data);
              v47_acc += ((v51_data[3]) * v21_data);
              v47_acc += ((v51_data[4]) * v22_data);
              v47_acc += ((v51_data[5]) * v23_data);
              v47_acc += ((v51_data[6]) * v24_data);
              v47_acc += ((v51_data[7]) * v25_data);
              v47_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v68_acc{};
              tensorforge::intel_esimd::simd<float, 16> v72_data;
              v72_data.copy_from(s0 + (16_i32));
              v68_acc += ((v72_data[0]) * v18_data);
              v68_acc += ((v72_data[1]) * v19_data);
              v68_acc += ((v72_data[2]) * v20_data);
              v68_acc += ((v72_data[3]) * v21_data);
              v68_acc += ((v72_data[4]) * v22_data);
              v68_acc += ((v72_data[5]) * v23_data);
              v68_acc += ((v72_data[6]) * v24_data);
              v68_acc += ((v72_data[7]) * v25_data);
              v68_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v89_acc{};
              tensorforge::intel_esimd::simd<float, 16> v93_data;
              v93_data.copy_from(s0 + (24_i32));
              v89_acc += ((v93_data[0]) * v18_data);
              v89_acc += ((v93_data[1]) * v19_data);
              v89_acc += ((v93_data[2]) * v20_data);
              v89_acc += ((v93_data[3]) * v21_data);
              v89_acc += ((v93_data[4]) * v22_data);
              v89_acc += ((v93_data[5]) * v23_data);
              v89_acc += ((v93_data[6]) * v24_data);
              v89_acc += ((v93_data[7]) * v25_data);
              v89_acc.copy_to(r1 + (48));
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v111_i1 = 0; v111_i1 < 4; ++v111_i1) {
                tensorforge::intel_esimd::simd<float, 8> v114_data;
                v114_data.copy_from(r1 + ((v111_i1 * 16)));
                v114_data.copy_to(s1 + ((v111_i1 * 8)));
              }
              float* __restrict__ s2 = &localShrMem0[0];
              // s2 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v120_ld;
              v120_ld.copy_from(glb_m2 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v120_ld.copy_to(s2 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              // wait(s2 = load{g>s}(glb_m2[0, 1]));
              float r2[64]{};
              // r2 = +(r0 * s2) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              float ir2[64]{};
              tensorforge::intel_esimd::simd<float, 16> v131_acc{};
              tensorforge::intel_esimd::simd<float, 16> v135_data;
              v135_data.copy_from(s2 + (0_i32));
              v131_acc += ((v135_data[0]) * v18_data);
              v131_acc += ((v135_data[1]) * v19_data);
              v131_acc += ((v135_data[2]) * v20_data);
              v131_acc += ((v135_data[3]) * v21_data);
              v131_acc += ((v135_data[4]) * v22_data);
              v131_acc += ((v135_data[5]) * v23_data);
              v131_acc += ((v135_data[6]) * v24_data);
              v131_acc += ((v135_data[7]) * v25_data);
              v131_acc.copy_to(ir2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v152_acc{};
              tensorforge::intel_esimd::simd<float, 16> v156_data;
              v156_data.copy_from(s2 + (8_i32));
              v152_acc += ((v156_data[0]) * v18_data);
              v152_acc += ((v156_data[1]) * v19_data);
              v152_acc += ((v156_data[2]) * v20_data);
              v152_acc += ((v156_data[3]) * v21_data);
              v152_acc += ((v156_data[4]) * v22_data);
              v152_acc += ((v156_data[5]) * v23_data);
              v152_acc += ((v156_data[6]) * v24_data);
              v152_acc += ((v156_data[7]) * v25_data);
              v152_acc.copy_to(ir2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v173_acc{};
              tensorforge::intel_esimd::simd<float, 16> v177_data;
              v177_data.copy_from(s2 + (16_i32));
              v173_acc += ((v177_data[0]) * v18_data);
              v173_acc += ((v177_data[1]) * v19_data);
              v173_acc += ((v177_data[2]) * v20_data);
              v173_acc += ((v177_data[3]) * v21_data);
              v173_acc += ((v177_data[4]) * v22_data);
              v173_acc += ((v177_data[5]) * v23_data);
              v173_acc += ((v177_data[6]) * v24_data);
              v173_acc += ((v177_data[7]) * v25_data);
              v173_acc.copy_to(ir2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v194_acc{};
              tensorforge::intel_esimd::simd<float, 16> v198_data;
              v198_data.copy_from(s2 + (24_i32));
              v194_acc += ((v198_data[0]) * v18_data);
              v194_acc += ((v198_data[1]) * v19_data);
              v194_acc += ((v198_data[2]) * v20_data);
              v194_acc += ((v198_data[3]) * v21_data);
              v194_acc += ((v198_data[4]) * v22_data);
              v194_acc += ((v198_data[5]) * v23_data);
              v194_acc += ((v198_data[6]) * v24_data);
              v194_acc += ((v198_data[7]) * v25_data);
              v194_acc.copy_to(ir2 + (48));
              #pragma unroll
              for (int32_t v215_n1 = 0; v215_n1 < 4; ++v215_n1) {
                int32_t v216_a = v215_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v218_data;
                v218_data.copy_from(ir2 + (v216_a));
                v218_data.copy_to(r2 + (v216_a));
              }
              // s1 = store{r>s}(localShrMem0, r2);
              #pragma unroll
              for (int32_t v221_i1 = 0; v221_i1 < 4; ++v221_i1) {
                tensorforge::intel_esimd::simd<float, 8> v224_data;
                v224_data.copy_from(r2 + ((v221_i1 * 16)));
                v224_data.copy_to(s1 + (((v221_i1 + 4) * 8)));
              }
              // glb_m3 = abs(s1)
              #pragma unroll
              for (int32_t v230_k1 = 0; v230_k1 < 8; ++v230_k1) {
                int32_t v233_a = v230_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v235_data;
                v235_data.copy_from(s1 + (v233_a));
                (tensorforge::intel_esimd::abs(v235_data)).copy_to(glb_m3 + (v233_a));
              }
            }
          }
        }
      });
    }
  });
}

