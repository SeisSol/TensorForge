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
              tensorforge::intel_esimd::simd<float, 16> v35_lin;
              v35_lin.copy_from(s0 + (0 + item.get_local_id(0) * 1));
              float v36_bc = v35_lin[0];
              v34_acc += (v36_bc * v28_data);
              float v38_bc = v35_lin[1];
              v34_acc += (v38_bc * v29_data);
              float v40_bc = v35_lin[2];
              v34_acc += (v40_bc * v30_data);
              float v42_bc = v35_lin[3];
              v34_acc += (v42_bc * v31_data);
              float v44_bc = v35_lin[4];
              v34_acc += (v44_bc * v32_data);
              float v46_bc = v35_lin[5];
              v34_acc += (v46_bc * v33_data);
              v34_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v48_acc{};
              v48_acc += (v36_bc * v28_data);
              v48_acc += (v38_bc * v29_data);
              v48_acc += (v40_bc * v30_data);
              v48_acc += (v42_bc * v31_data);
              v48_acc += (v44_bc * v32_data);
              v48_acc += (v46_bc * v33_data);
              v48_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v62_acc{};
              v62_acc += (v36_bc * v28_data);
              v62_acc += (v38_bc * v29_data);
              v62_acc += (v40_bc * v30_data);
              v62_acc += (v42_bc * v31_data);
              v62_acc += (v44_bc * v32_data);
              v62_acc += (v46_bc * v33_data);
              v62_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v76_acc{};
              v76_acc += (v36_bc * v28_data);
              v76_acc += (v38_bc * v29_data);
              v76_acc += (v40_bc * v30_data);
              v76_acc += (v42_bc * v31_data);
              v76_acc += (v44_bc * v32_data);
              v76_acc += (v46_bc * v33_data);
              v76_acc.copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v90_acc{};
              v90_acc += (v36_bc * v28_data);
              v90_acc += (v38_bc * v29_data);
              v90_acc += (v40_bc * v30_data);
              v90_acc += (v42_bc * v31_data);
              v90_acc += (v44_bc * v32_data);
              v90_acc += (v46_bc * v33_data);
              v90_acc.copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v104_acc{};
              v104_acc += (v36_bc * v28_data);
              v104_acc += (v38_bc * v29_data);
              v104_acc += (v40_bc * v30_data);
              v104_acc += (v42_bc * v31_data);
              v104_acc += (v44_bc * v32_data);
              v104_acc += (v46_bc * v33_data);
              v104_acc.copy_to(r1 + (80));
              // wait(r2 = load{g>r}(glb_m3););
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v119_i1 = 0; v119_i1 < 6; ++v119_i1) {
                tensorforge::intel_esimd::simd<float, 12> v122_data;
                v122_data.copy_from(r1 + ((v119_i1 * 16)));
                v122_data.copy_to(s1 + ((v119_i1 * 12)));
              }
              float r3[96]{};
              // r3 = +(r2 * s1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir3[96]{};
              tensorforge::intel_esimd::simd<float, 16> v129_data;
              v129_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v130_data;
              v130_data.copy_from(r2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v131_data;
              v131_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v132_data;
              v132_data.copy_from(r2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v133_data;
              v133_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v134_data;
              v134_data.copy_from(r2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v135_data;
              v135_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v136_data;
              v136_data.copy_from(r2 + (112));
              tensorforge::intel_esimd::simd<float, 16> v137_data;
              v137_data.copy_from(r2 + (128));
              tensorforge::intel_esimd::simd<float, 16> v138_data;
              v138_data.copy_from(r2 + (144));
              tensorforge::intel_esimd::simd<float, 16> v139_data;
              v139_data.copy_from(r2 + (160));
              tensorforge::intel_esimd::simd<float, 16> v140_data;
              v140_data.copy_from(r2 + (176));
              tensorforge::intel_esimd::simd<float, 16> v141_acc{};
              tensorforge::intel_esimd::simd<float, 12> v142_lin;
              v142_lin.copy_from(s1 + (0 + item.get_local_id(0) * 1));
              float v143_bc = v142_lin[0];
              v141_acc += (v143_bc * v129_data);
              float v145_bc = v142_lin[1];
              v141_acc += (v145_bc * v130_data);
              float v147_bc = v142_lin[2];
              v141_acc += (v147_bc * v131_data);
              float v149_bc = v142_lin[3];
              v141_acc += (v149_bc * v132_data);
              float v151_bc = v142_lin[4];
              v141_acc += (v151_bc * v133_data);
              float v153_bc = v142_lin[5];
              v141_acc += (v153_bc * v134_data);
              float v155_bc = v142_lin[6];
              v141_acc += (v155_bc * v135_data);
              float v157_bc = v142_lin[7];
              v141_acc += (v157_bc * v136_data);
              float v159_bc = v142_lin[8];
              v141_acc += (v159_bc * v137_data);
              float v161_bc = v142_lin[9];
              v141_acc += (v161_bc * v138_data);
              float v163_bc = v142_lin[10];
              v141_acc += (v163_bc * v139_data);
              float v165_bc = v142_lin[11];
              v141_acc += (v165_bc * v140_data);
              v141_acc.copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v167_acc{};
              v167_acc += (v143_bc * v129_data);
              v167_acc += (v145_bc * v130_data);
              v167_acc += (v147_bc * v131_data);
              v167_acc += (v149_bc * v132_data);
              v167_acc += (v151_bc * v133_data);
              v167_acc += (v153_bc * v134_data);
              v167_acc += (v155_bc * v135_data);
              v167_acc += (v157_bc * v136_data);
              v167_acc += (v159_bc * v137_data);
              v167_acc += (v161_bc * v138_data);
              v167_acc += (v163_bc * v139_data);
              v167_acc += (v165_bc * v140_data);
              v167_acc.copy_to(ir3 + (16));
              tensorforge::intel_esimd::simd<float, 16> v193_acc{};
              v193_acc += (v143_bc * v129_data);
              v193_acc += (v145_bc * v130_data);
              v193_acc += (v147_bc * v131_data);
              v193_acc += (v149_bc * v132_data);
              v193_acc += (v151_bc * v133_data);
              v193_acc += (v153_bc * v134_data);
              v193_acc += (v155_bc * v135_data);
              v193_acc += (v157_bc * v136_data);
              v193_acc += (v159_bc * v137_data);
              v193_acc += (v161_bc * v138_data);
              v193_acc += (v163_bc * v139_data);
              v193_acc += (v165_bc * v140_data);
              v193_acc.copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v219_acc{};
              v219_acc += (v143_bc * v129_data);
              v219_acc += (v145_bc * v130_data);
              v219_acc += (v147_bc * v131_data);
              v219_acc += (v149_bc * v132_data);
              v219_acc += (v151_bc * v133_data);
              v219_acc += (v153_bc * v134_data);
              v219_acc += (v155_bc * v135_data);
              v219_acc += (v157_bc * v136_data);
              v219_acc += (v159_bc * v137_data);
              v219_acc += (v161_bc * v138_data);
              v219_acc += (v163_bc * v139_data);
              v219_acc += (v165_bc * v140_data);
              v219_acc.copy_to(ir3 + (48));
              tensorforge::intel_esimd::simd<float, 16> v245_acc{};
              v245_acc += (v143_bc * v129_data);
              v245_acc += (v145_bc * v130_data);
              v245_acc += (v147_bc * v131_data);
              v245_acc += (v149_bc * v132_data);
              v245_acc += (v151_bc * v133_data);
              v245_acc += (v153_bc * v134_data);
              v245_acc += (v155_bc * v135_data);
              v245_acc += (v157_bc * v136_data);
              v245_acc += (v159_bc * v137_data);
              v245_acc += (v161_bc * v138_data);
              v245_acc += (v163_bc * v139_data);
              v245_acc += (v165_bc * v140_data);
              v245_acc.copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v271_acc{};
              v271_acc += (v143_bc * v129_data);
              v271_acc += (v145_bc * v130_data);
              v271_acc += (v147_bc * v131_data);
              v271_acc += (v149_bc * v132_data);
              v271_acc += (v151_bc * v133_data);
              v271_acc += (v153_bc * v134_data);
              v271_acc += (v155_bc * v135_data);
              v271_acc += (v157_bc * v136_data);
              v271_acc += (v159_bc * v137_data);
              v271_acc += (v161_bc * v138_data);
              v271_acc += (v163_bc * v139_data);
              v271_acc += (v165_bc * v140_data);
              v271_acc.copy_to(ir3 + (80));
              #pragma unroll
              for (int32_t v297_n1 = 0; v297_n1 < 6; ++v297_n1) {
                int32_t v298_a = v297_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v300_data;
                v300_data.copy_from(ir3 + (v298_a));
                v300_data.copy_to(r3 + (v298_a));
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v303_i1 = 0; v303_i1 < 6; ++v303_i1) {
                tensorforge::intel_esimd::simd<float, 12> v306_data;
                v306_data.copy_from(r3 + ((v303_i1 * 16)));
                v306_data.copy_to(glb_m2 + ((v303_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

