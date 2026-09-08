// === base name ===
kernel_609dd06e89

// === header ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_609dd06e89(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_609dd06e89(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // m3 8×8(8×8) {0..8}×{0..8} strided
        // m4 8×8(8×8) {0..8}×{0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] += m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m3 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
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
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[batchId0 * 64 + 0 + m4_extraOffset];
              float r0[128]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v8_i1 = 0; v8_i1 < 8; ++v8_i1) {
                tensorforge::intel_esimd::simd<float, 8> v13_data;
                v13_data.copy_from(glb_m0 + ((v8_i1 * 8)));
                v13_data.copy_to(r0 + ((v8_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v17_ld;
              v17_ld.copy_from(glb_m1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v17_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m0););
              float r2[128]{};
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 8; ++v19_i1) {
                tensorforge::intel_esimd::simd<float, 8> v24_data;
                v24_data.copy_from(glb_m2 + ((v19_i1 * 8)));
                v24_data.copy_to(r2 + ((v19_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
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
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v36_acc{};
              tensorforge::intel_esimd::simd<float, 16> v37_lin;
              v37_lin.copy_from(s0 + (0 + item.get_local_id(0) * 1));
              float v38_bc = v37_lin[0];
              v36_acc += (v38_bc * v28_data);
              float v40_bc = v37_lin[1];
              v36_acc += (v40_bc * v29_data);
              float v42_bc = v37_lin[2];
              v36_acc += (v42_bc * v30_data);
              float v44_bc = v37_lin[3];
              v36_acc += (v44_bc * v31_data);
              float v46_bc = v37_lin[4];
              v36_acc += (v46_bc * v32_data);
              float v48_bc = v37_lin[5];
              v36_acc += (v48_bc * v33_data);
              float v50_bc = v37_lin[6];
              v36_acc += (v50_bc * v34_data);
              float v52_bc = v37_lin[7];
              v36_acc += (v52_bc * v35_data);
              v36_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v54_acc{};
              v54_acc += (v38_bc * v28_data);
              v54_acc += (v40_bc * v29_data);
              v54_acc += (v42_bc * v30_data);
              v54_acc += (v44_bc * v31_data);
              v54_acc += (v46_bc * v32_data);
              v54_acc += (v48_bc * v33_data);
              v54_acc += (v50_bc * v34_data);
              v54_acc += (v52_bc * v35_data);
              v54_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v72_acc{};
              v72_acc += (v38_bc * v28_data);
              v72_acc += (v40_bc * v29_data);
              v72_acc += (v42_bc * v30_data);
              v72_acc += (v44_bc * v31_data);
              v72_acc += (v46_bc * v32_data);
              v72_acc += (v48_bc * v33_data);
              v72_acc += (v50_bc * v34_data);
              v72_acc += (v52_bc * v35_data);
              v72_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v90_acc{};
              v90_acc += (v38_bc * v28_data);
              v90_acc += (v40_bc * v29_data);
              v90_acc += (v42_bc * v30_data);
              v90_acc += (v44_bc * v31_data);
              v90_acc += (v46_bc * v32_data);
              v90_acc += (v48_bc * v33_data);
              v90_acc += (v50_bc * v34_data);
              v90_acc += (v52_bc * v35_data);
              v90_acc.copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v108_acc{};
              v108_acc += (v38_bc * v28_data);
              v108_acc += (v40_bc * v29_data);
              v108_acc += (v42_bc * v30_data);
              v108_acc += (v44_bc * v31_data);
              v108_acc += (v46_bc * v32_data);
              v108_acc += (v48_bc * v33_data);
              v108_acc += (v50_bc * v34_data);
              v108_acc += (v52_bc * v35_data);
              v108_acc.copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v126_acc{};
              v126_acc += (v38_bc * v28_data);
              v126_acc += (v40_bc * v29_data);
              v126_acc += (v42_bc * v30_data);
              v126_acc += (v44_bc * v31_data);
              v126_acc += (v46_bc * v32_data);
              v126_acc += (v48_bc * v33_data);
              v126_acc += (v50_bc * v34_data);
              v126_acc += (v52_bc * v35_data);
              v126_acc.copy_to(r1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v144_acc{};
              v144_acc += (v38_bc * v28_data);
              v144_acc += (v40_bc * v29_data);
              v144_acc += (v42_bc * v30_data);
              v144_acc += (v44_bc * v31_data);
              v144_acc += (v46_bc * v32_data);
              v144_acc += (v48_bc * v33_data);
              v144_acc += (v50_bc * v34_data);
              v144_acc += (v52_bc * v35_data);
              v144_acc.copy_to(r1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v162_acc{};
              v162_acc += (v38_bc * v28_data);
              v162_acc += (v40_bc * v29_data);
              v162_acc += (v42_bc * v30_data);
              v162_acc += (v44_bc * v31_data);
              v162_acc += (v46_bc * v32_data);
              v162_acc += (v48_bc * v33_data);
              v162_acc += (v50_bc * v34_data);
              v162_acc += (v52_bc * v35_data);
              v162_acc.copy_to(r1 + (112));
              float* __restrict__ s2 = &localShrMem0[0];
              // s2 = load{g>s}(glb_m3[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v181_ld;
              v181_ld.copy_from(glb_m3 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v181_ld.copy_to(s2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              // wait(r2 = load{g>r}(glb_m2););
              // wait(s2 = load{g>s}(glb_m3[0, 1]));
              float r3[128]{};
              // r3 = +(r2 * s2) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir3[128]{};
              tensorforge::intel_esimd::simd<float, 16> v184_data;
              v184_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v185_data;
              v185_data.copy_from(r2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v186_data;
              v186_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v187_data;
              v187_data.copy_from(r2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v188_data;
              v188_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v189_data;
              v189_data.copy_from(r2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v190_data;
              v190_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v191_data;
              v191_data.copy_from(r2 + (112));
              tensorforge::intel_esimd::simd<float, 16> v192_acc{};
              tensorforge::intel_esimd::simd<float, 16> v193_lin;
              v193_lin.copy_from(s2 + (0 + item.get_local_id(0) * 1));
              float v194_bc = v193_lin[0];
              v192_acc += (v194_bc * v184_data);
              float v196_bc = v193_lin[1];
              v192_acc += (v196_bc * v185_data);
              float v198_bc = v193_lin[2];
              v192_acc += (v198_bc * v186_data);
              float v200_bc = v193_lin[3];
              v192_acc += (v200_bc * v187_data);
              float v202_bc = v193_lin[4];
              v192_acc += (v202_bc * v188_data);
              float v204_bc = v193_lin[5];
              v192_acc += (v204_bc * v189_data);
              float v206_bc = v193_lin[6];
              v192_acc += (v206_bc * v190_data);
              float v208_bc = v193_lin[7];
              v192_acc += (v208_bc * v191_data);
              v192_acc.copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v210_acc{};
              v210_acc += (v194_bc * v184_data);
              v210_acc += (v196_bc * v185_data);
              v210_acc += (v198_bc * v186_data);
              v210_acc += (v200_bc * v187_data);
              v210_acc += (v202_bc * v188_data);
              v210_acc += (v204_bc * v189_data);
              v210_acc += (v206_bc * v190_data);
              v210_acc += (v208_bc * v191_data);
              v210_acc.copy_to(ir3 + (16));
              tensorforge::intel_esimd::simd<float, 16> v228_acc{};
              v228_acc += (v194_bc * v184_data);
              v228_acc += (v196_bc * v185_data);
              v228_acc += (v198_bc * v186_data);
              v228_acc += (v200_bc * v187_data);
              v228_acc += (v202_bc * v188_data);
              v228_acc += (v204_bc * v189_data);
              v228_acc += (v206_bc * v190_data);
              v228_acc += (v208_bc * v191_data);
              v228_acc.copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v246_acc{};
              v246_acc += (v194_bc * v184_data);
              v246_acc += (v196_bc * v185_data);
              v246_acc += (v198_bc * v186_data);
              v246_acc += (v200_bc * v187_data);
              v246_acc += (v202_bc * v188_data);
              v246_acc += (v204_bc * v189_data);
              v246_acc += (v206_bc * v190_data);
              v246_acc += (v208_bc * v191_data);
              v246_acc.copy_to(ir3 + (48));
              tensorforge::intel_esimd::simd<float, 16> v264_acc{};
              v264_acc += (v194_bc * v184_data);
              v264_acc += (v196_bc * v185_data);
              v264_acc += (v198_bc * v186_data);
              v264_acc += (v200_bc * v187_data);
              v264_acc += (v202_bc * v188_data);
              v264_acc += (v204_bc * v189_data);
              v264_acc += (v206_bc * v190_data);
              v264_acc += (v208_bc * v191_data);
              v264_acc.copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v282_acc{};
              v282_acc += (v194_bc * v184_data);
              v282_acc += (v196_bc * v185_data);
              v282_acc += (v198_bc * v186_data);
              v282_acc += (v200_bc * v187_data);
              v282_acc += (v202_bc * v188_data);
              v282_acc += (v204_bc * v189_data);
              v282_acc += (v206_bc * v190_data);
              v282_acc += (v208_bc * v191_data);
              v282_acc.copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 16> v300_acc{};
              v300_acc += (v194_bc * v184_data);
              v300_acc += (v196_bc * v185_data);
              v300_acc += (v198_bc * v186_data);
              v300_acc += (v200_bc * v187_data);
              v300_acc += (v202_bc * v188_data);
              v300_acc += (v204_bc * v189_data);
              v300_acc += (v206_bc * v190_data);
              v300_acc += (v208_bc * v191_data);
              v300_acc.copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 16> v318_acc{};
              v318_acc += (v194_bc * v184_data);
              v318_acc += (v196_bc * v185_data);
              v318_acc += (v198_bc * v186_data);
              v318_acc += (v200_bc * v187_data);
              v318_acc += (v202_bc * v188_data);
              v318_acc += (v204_bc * v189_data);
              v318_acc += (v206_bc * v190_data);
              v318_acc += (v208_bc * v191_data);
              v318_acc.copy_to(ir3 + (112));
              #pragma unroll
              for (int32_t v336_n1 = 0; v336_n1 < 8; ++v336_n1) {
                int32_t v337_a = v336_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v339_data;
                v339_data.copy_from(ir3 + (v337_a));
                tensorforge::intel_esimd::simd<float, 8> v342_data;
                v342_data.copy_from(r1 + (v337_a));
                (v342_data + v339_data).copy_to(r3 + (v337_a));
              }
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v347_i1 = 0; v347_i1 < 8; ++v347_i1) {
                tensorforge::intel_esimd::simd<float, 8> v350_data;
                v350_data.copy_from(r3 + ((v347_i1 * 16)));
                v350_data.copy_to(s1 + ((v347_i1 * 8)));
              }
              // glb_m4 = abs(s1)
              #pragma unroll
              for (int32_t v355_k1 = 0; v355_k1 < 8; ++v355_k1) {
                int32_t v358_a = v355_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v360_data;
                v360_data.copy_from(s1 + (v358_a));
                (tensorforge::intel_esimd::abs(v360_data)).copy_to(glb_m4 + (v358_a));
              }
            }
          }
        }
      });
    }
  });
}

