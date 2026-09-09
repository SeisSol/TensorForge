// === base name ===
kernel_07df6415d476a0ea

// === header ===
void launcher_kernel_07df6415d476a0ea(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_07df6415d476a0ea(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_07df6415d476a0ea(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_07df6415d476a0ea(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×16(12×16) {0..12}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[144 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[128];
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v11_i1 = 0; v11_i1 < 16; ++v11_i1) {
                tensorforge::intel_esimd::simd<float, 12> v16_data;
                v16_data.copy_from(glb_m1 + ((v11_i1 * 12)));
                v16_data.copy_to(r0 + ((v11_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v19_ld;
              v19_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v19_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v20_ld;
              v20_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v20_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              float r1[128]{};
              // r1 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v22_i1 = 0; v22_i1 < 8; ++v22_i1) {
                tensorforge::intel_esimd::simd<float, 12> v27_data;
                v27_data.copy_from(glb_m0 + ((v22_i1 * 12)));
                v27_data.copy_to(r1 + ((v22_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              // wait(r1 = load{g>r}(glb_m0););
              float r2[128]{};
              // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir2[128]{};
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v46_data;
              v46_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v48_acc{};
              tensorforge::intel_esimd::simd<float, 16> v52_data;
              v52_data.copy_from(s0 + (0_i32));
              v48_acc += ((v52_data[0]) * v32_data);
              v48_acc += ((v52_data[1]) * v33_data);
              v48_acc += ((v52_data[2]) * v34_data);
              v48_acc += ((v52_data[3]) * v35_data);
              v48_acc += ((v52_data[4]) * v36_data);
              v48_acc += ((v52_data[5]) * v37_data);
              v48_acc += ((v52_data[6]) * v38_data);
              v48_acc += ((v52_data[7]) * v39_data);
              v48_acc += ((v52_data[8]) * v40_data);
              v48_acc += ((v52_data[9]) * v41_data);
              v48_acc += ((v52_data[10]) * v42_data);
              v48_acc += ((v52_data[11]) * v43_data);
              v48_acc += ((v52_data[12]) * v44_data);
              v48_acc += ((v52_data[13]) * v45_data);
              v48_acc += ((v52_data[14]) * v46_data);
              v48_acc += ((v52_data[15]) * v47_data);
              v48_acc.copy_to(ir2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v85_acc{};
              tensorforge::intel_esimd::simd<float, 16> v89_data;
              v89_data.copy_from(s0 + (16_i32));
              v85_acc += ((v89_data[0]) * v32_data);
              v85_acc += ((v89_data[1]) * v33_data);
              v85_acc += ((v89_data[2]) * v34_data);
              v85_acc += ((v89_data[3]) * v35_data);
              v85_acc += ((v89_data[4]) * v36_data);
              v85_acc += ((v89_data[5]) * v37_data);
              v85_acc += ((v89_data[6]) * v38_data);
              v85_acc += ((v89_data[7]) * v39_data);
              v85_acc += ((v89_data[8]) * v40_data);
              v85_acc += ((v89_data[9]) * v41_data);
              v85_acc += ((v89_data[10]) * v42_data);
              v85_acc += ((v89_data[11]) * v43_data);
              v85_acc += ((v89_data[12]) * v44_data);
              v85_acc += ((v89_data[13]) * v45_data);
              v85_acc += ((v89_data[14]) * v46_data);
              v85_acc += ((v89_data[15]) * v47_data);
              v85_acc.copy_to(ir2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v122_acc{};
              tensorforge::intel_esimd::simd<float, 16> v126_data;
              v126_data.copy_from(s0 + (32_i32));
              v122_acc += ((v126_data[0]) * v32_data);
              v122_acc += ((v126_data[1]) * v33_data);
              v122_acc += ((v126_data[2]) * v34_data);
              v122_acc += ((v126_data[3]) * v35_data);
              v122_acc += ((v126_data[4]) * v36_data);
              v122_acc += ((v126_data[5]) * v37_data);
              v122_acc += ((v126_data[6]) * v38_data);
              v122_acc += ((v126_data[7]) * v39_data);
              v122_acc += ((v126_data[8]) * v40_data);
              v122_acc += ((v126_data[9]) * v41_data);
              v122_acc += ((v126_data[10]) * v42_data);
              v122_acc += ((v126_data[11]) * v43_data);
              v122_acc += ((v126_data[12]) * v44_data);
              v122_acc += ((v126_data[13]) * v45_data);
              v122_acc += ((v126_data[14]) * v46_data);
              v122_acc += ((v126_data[15]) * v47_data);
              v122_acc.copy_to(ir2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v159_acc{};
              tensorforge::intel_esimd::simd<float, 16> v163_data;
              v163_data.copy_from(s0 + (48_i32));
              v159_acc += ((v163_data[0]) * v32_data);
              v159_acc += ((v163_data[1]) * v33_data);
              v159_acc += ((v163_data[2]) * v34_data);
              v159_acc += ((v163_data[3]) * v35_data);
              v159_acc += ((v163_data[4]) * v36_data);
              v159_acc += ((v163_data[5]) * v37_data);
              v159_acc += ((v163_data[6]) * v38_data);
              v159_acc += ((v163_data[7]) * v39_data);
              v159_acc += ((v163_data[8]) * v40_data);
              v159_acc += ((v163_data[9]) * v41_data);
              v159_acc += ((v163_data[10]) * v42_data);
              v159_acc += ((v163_data[11]) * v43_data);
              v159_acc += ((v163_data[12]) * v44_data);
              v159_acc += ((v163_data[13]) * v45_data);
              v159_acc += ((v163_data[14]) * v46_data);
              v159_acc += ((v163_data[15]) * v47_data);
              v159_acc.copy_to(ir2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v196_acc{};
              tensorforge::intel_esimd::simd<float, 16> v200_data;
              v200_data.copy_from(s0 + (64_i32));
              v196_acc += ((v200_data[0]) * v32_data);
              v196_acc += ((v200_data[1]) * v33_data);
              v196_acc += ((v200_data[2]) * v34_data);
              v196_acc += ((v200_data[3]) * v35_data);
              v196_acc += ((v200_data[4]) * v36_data);
              v196_acc += ((v200_data[5]) * v37_data);
              v196_acc += ((v200_data[6]) * v38_data);
              v196_acc += ((v200_data[7]) * v39_data);
              v196_acc += ((v200_data[8]) * v40_data);
              v196_acc += ((v200_data[9]) * v41_data);
              v196_acc += ((v200_data[10]) * v42_data);
              v196_acc += ((v200_data[11]) * v43_data);
              v196_acc += ((v200_data[12]) * v44_data);
              v196_acc += ((v200_data[13]) * v45_data);
              v196_acc += ((v200_data[14]) * v46_data);
              v196_acc += ((v200_data[15]) * v47_data);
              v196_acc.copy_to(ir2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v233_acc{};
              tensorforge::intel_esimd::simd<float, 16> v237_data;
              v237_data.copy_from(s0 + (80_i32));
              v233_acc += ((v237_data[0]) * v32_data);
              v233_acc += ((v237_data[1]) * v33_data);
              v233_acc += ((v237_data[2]) * v34_data);
              v233_acc += ((v237_data[3]) * v35_data);
              v233_acc += ((v237_data[4]) * v36_data);
              v233_acc += ((v237_data[5]) * v37_data);
              v233_acc += ((v237_data[6]) * v38_data);
              v233_acc += ((v237_data[7]) * v39_data);
              v233_acc += ((v237_data[8]) * v40_data);
              v233_acc += ((v237_data[9]) * v41_data);
              v233_acc += ((v237_data[10]) * v42_data);
              v233_acc += ((v237_data[11]) * v43_data);
              v233_acc += ((v237_data[12]) * v44_data);
              v233_acc += ((v237_data[13]) * v45_data);
              v233_acc += ((v237_data[14]) * v46_data);
              v233_acc += ((v237_data[15]) * v47_data);
              v233_acc.copy_to(ir2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v270_acc{};
              tensorforge::intel_esimd::simd<float, 16> v274_data;
              v274_data.copy_from(s0 + (96_i32));
              v270_acc += ((v274_data[0]) * v32_data);
              v270_acc += ((v274_data[1]) * v33_data);
              v270_acc += ((v274_data[2]) * v34_data);
              v270_acc += ((v274_data[3]) * v35_data);
              v270_acc += ((v274_data[4]) * v36_data);
              v270_acc += ((v274_data[5]) * v37_data);
              v270_acc += ((v274_data[6]) * v38_data);
              v270_acc += ((v274_data[7]) * v39_data);
              v270_acc += ((v274_data[8]) * v40_data);
              v270_acc += ((v274_data[9]) * v41_data);
              v270_acc += ((v274_data[10]) * v42_data);
              v270_acc += ((v274_data[11]) * v43_data);
              v270_acc += ((v274_data[12]) * v44_data);
              v270_acc += ((v274_data[13]) * v45_data);
              v270_acc += ((v274_data[14]) * v46_data);
              v270_acc += ((v274_data[15]) * v47_data);
              v270_acc.copy_to(ir2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v307_acc{};
              tensorforge::intel_esimd::simd<float, 16> v311_data;
              v311_data.copy_from(s0 + (112_i32));
              v307_acc += ((v311_data[0]) * v32_data);
              v307_acc += ((v311_data[1]) * v33_data);
              v307_acc += ((v311_data[2]) * v34_data);
              v307_acc += ((v311_data[3]) * v35_data);
              v307_acc += ((v311_data[4]) * v36_data);
              v307_acc += ((v311_data[5]) * v37_data);
              v307_acc += ((v311_data[6]) * v38_data);
              v307_acc += ((v311_data[7]) * v39_data);
              v307_acc += ((v311_data[8]) * v40_data);
              v307_acc += ((v311_data[9]) * v41_data);
              v307_acc += ((v311_data[10]) * v42_data);
              v307_acc += ((v311_data[11]) * v43_data);
              v307_acc += ((v311_data[12]) * v44_data);
              v307_acc += ((v311_data[13]) * v45_data);
              v307_acc += ((v311_data[14]) * v46_data);
              v307_acc += ((v311_data[15]) * v47_data);
              v307_acc.copy_to(ir2 + (112));
              #pragma unroll
              for (int32_t v344_n1 = 0; v344_n1 < 8; ++v344_n1) {
                int32_t v345_a = v344_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v347_data;
                v347_data.copy_from(ir2 + (v345_a));
                tensorforge::intel_esimd::simd<float, 12> v350_data;
                v350_data.copy_from(r1 + (v345_a));
                (v350_data + v347_data).copy_to(r2 + (v345_a));
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v354_i1 = 0; v354_i1 < 8; ++v354_i1) {
                tensorforge::intel_esimd::simd<float, 12> v357_data;
                v357_data.copy_from(r2 + ((v354_i1 * 16)));
                v357_data.copy_to(glb_m0 + ((v354_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

