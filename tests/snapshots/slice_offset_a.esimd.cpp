// === base name ===
kernel_f61651fe59

// === header ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f61651fe59(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f61651fe59(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 32×16(12×16) {4..16}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(12×16) {4..16}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
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
              tensorforge::intel_esimd::simd<float, 16> v11_data;
              v11_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v17_data;
              v17_data.copy_from(glb_m1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(glb_m1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(glb_m1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(glb_m1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(glb_m1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(glb_m1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v53_data;
              v53_data.copy_from(glb_m1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v59_data;
              v59_data.copy_from(glb_m1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data;
              v65_data.copy_from(glb_m1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v71_data;
              v71_data.copy_from(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v77_data;
              v77_data.copy_from(glb_m1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v83_data;
              v83_data.copy_from(glb_m1 + (144_i32));
              tensorforge::intel_esimd::simd<float, 16> v89_data;
              v89_data.copy_from(glb_m1 + (156_i32));
              tensorforge::intel_esimd::simd<float, 16> v95_data;
              v95_data.copy_from(glb_m1 + (168_i32));
              tensorforge::intel_esimd::simd<float, 16> v101_data;
              v101_data.copy_from(glb_m1 + (180_i32));
              tensorforge::intel_esimd::simd<float, 16> v102_acc{};
              tensorforge::intel_esimd::simd<float, 16> v109_data;
              v109_data.copy_from(s0 + ((0_i32 ^ ((0_i32 >> 4) & 15))));
              v102_acc += ((v109_data[0]) * v11_data);
              v102_acc += ((v109_data[1]) * v17_data);
              v102_acc += ((v109_data[2]) * v23_data);
              v102_acc += ((v109_data[3]) * v29_data);
              v102_acc += ((v109_data[4]) * v35_data);
              v102_acc += ((v109_data[5]) * v41_data);
              v102_acc += ((v109_data[6]) * v47_data);
              v102_acc += ((v109_data[7]) * v53_data);
              v102_acc += ((v109_data[8]) * v59_data);
              v102_acc += ((v109_data[9]) * v65_data);
              v102_acc += ((v109_data[10]) * v71_data);
              v102_acc += ((v109_data[11]) * v77_data);
              v102_acc += ((v109_data[12]) * v83_data);
              v102_acc += ((v109_data[13]) * v89_data);
              v102_acc += ((v109_data[14]) * v95_data);
              v102_acc += ((v109_data[15]) * v101_data);
              v102_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v142_acc{};
              tensorforge::intel_esimd::simd<float, 16> v149_data;
              v149_data.copy_from(s0 + ((16_i32 ^ ((16_i32 >> 4) & 15))));
              v142_acc += ((v149_data[0]) * v11_data);
              v142_acc += ((v149_data[1]) * v17_data);
              v142_acc += ((v149_data[2]) * v23_data);
              v142_acc += ((v149_data[3]) * v29_data);
              v142_acc += ((v149_data[4]) * v35_data);
              v142_acc += ((v149_data[5]) * v41_data);
              v142_acc += ((v149_data[6]) * v47_data);
              v142_acc += ((v149_data[7]) * v53_data);
              v142_acc += ((v149_data[8]) * v59_data);
              v142_acc += ((v149_data[9]) * v65_data);
              v142_acc += ((v149_data[10]) * v71_data);
              v142_acc += ((v149_data[11]) * v77_data);
              v142_acc += ((v149_data[12]) * v83_data);
              v142_acc += ((v149_data[13]) * v89_data);
              v142_acc += ((v149_data[14]) * v95_data);
              v142_acc += ((v149_data[15]) * v101_data);
              v142_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v182_acc{};
              tensorforge::intel_esimd::simd<float, 16> v189_data;
              v189_data.copy_from(s0 + ((32_i32 ^ ((32_i32 >> 4) & 15))));
              v182_acc += ((v189_data[0]) * v11_data);
              v182_acc += ((v189_data[1]) * v17_data);
              v182_acc += ((v189_data[2]) * v23_data);
              v182_acc += ((v189_data[3]) * v29_data);
              v182_acc += ((v189_data[4]) * v35_data);
              v182_acc += ((v189_data[5]) * v41_data);
              v182_acc += ((v189_data[6]) * v47_data);
              v182_acc += ((v189_data[7]) * v53_data);
              v182_acc += ((v189_data[8]) * v59_data);
              v182_acc += ((v189_data[9]) * v65_data);
              v182_acc += ((v189_data[10]) * v71_data);
              v182_acc += ((v189_data[11]) * v77_data);
              v182_acc += ((v189_data[12]) * v83_data);
              v182_acc += ((v189_data[13]) * v89_data);
              v182_acc += ((v189_data[14]) * v95_data);
              v182_acc += ((v189_data[15]) * v101_data);
              v182_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v222_acc{};
              tensorforge::intel_esimd::simd<float, 16> v229_data;
              v229_data.copy_from(s0 + ((48_i32 ^ ((48_i32 >> 4) & 15))));
              v222_acc += ((v229_data[0]) * v11_data);
              v222_acc += ((v229_data[1]) * v17_data);
              v222_acc += ((v229_data[2]) * v23_data);
              v222_acc += ((v229_data[3]) * v29_data);
              v222_acc += ((v229_data[4]) * v35_data);
              v222_acc += ((v229_data[5]) * v41_data);
              v222_acc += ((v229_data[6]) * v47_data);
              v222_acc += ((v229_data[7]) * v53_data);
              v222_acc += ((v229_data[8]) * v59_data);
              v222_acc += ((v229_data[9]) * v65_data);
              v222_acc += ((v229_data[10]) * v71_data);
              v222_acc += ((v229_data[11]) * v77_data);
              v222_acc += ((v229_data[12]) * v83_data);
              v222_acc += ((v229_data[13]) * v89_data);
              v222_acc += ((v229_data[14]) * v95_data);
              v222_acc += ((v229_data[15]) * v101_data);
              v222_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v262_acc{};
              tensorforge::intel_esimd::simd<float, 16> v269_data;
              v269_data.copy_from(s0 + ((64_i32 ^ ((64_i32 >> 4) & 15))));
              v262_acc += ((v269_data[0]) * v11_data);
              v262_acc += ((v269_data[1]) * v17_data);
              v262_acc += ((v269_data[2]) * v23_data);
              v262_acc += ((v269_data[3]) * v29_data);
              v262_acc += ((v269_data[4]) * v35_data);
              v262_acc += ((v269_data[5]) * v41_data);
              v262_acc += ((v269_data[6]) * v47_data);
              v262_acc += ((v269_data[7]) * v53_data);
              v262_acc += ((v269_data[8]) * v59_data);
              v262_acc += ((v269_data[9]) * v65_data);
              v262_acc += ((v269_data[10]) * v71_data);
              v262_acc += ((v269_data[11]) * v77_data);
              v262_acc += ((v269_data[12]) * v83_data);
              v262_acc += ((v269_data[13]) * v89_data);
              v262_acc += ((v269_data[14]) * v95_data);
              v262_acc += ((v269_data[15]) * v101_data);
              v262_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v302_acc{};
              tensorforge::intel_esimd::simd<float, 16> v309_data;
              v309_data.copy_from(s0 + ((80_i32 ^ ((80_i32 >> 4) & 15))));
              v302_acc += ((v309_data[0]) * v11_data);
              v302_acc += ((v309_data[1]) * v17_data);
              v302_acc += ((v309_data[2]) * v23_data);
              v302_acc += ((v309_data[3]) * v29_data);
              v302_acc += ((v309_data[4]) * v35_data);
              v302_acc += ((v309_data[5]) * v41_data);
              v302_acc += ((v309_data[6]) * v47_data);
              v302_acc += ((v309_data[7]) * v53_data);
              v302_acc += ((v309_data[8]) * v59_data);
              v302_acc += ((v309_data[9]) * v65_data);
              v302_acc += ((v309_data[10]) * v71_data);
              v302_acc += ((v309_data[11]) * v77_data);
              v302_acc += ((v309_data[12]) * v83_data);
              v302_acc += ((v309_data[13]) * v89_data);
              v302_acc += ((v309_data[14]) * v95_data);
              v302_acc += ((v309_data[15]) * v101_data);
              v302_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v342_acc{};
              tensorforge::intel_esimd::simd<float, 16> v349_data;
              v349_data.copy_from(s0 + ((96_i32 ^ ((96_i32 >> 4) & 15))));
              v342_acc += ((v349_data[0]) * v11_data);
              v342_acc += ((v349_data[1]) * v17_data);
              v342_acc += ((v349_data[2]) * v23_data);
              v342_acc += ((v349_data[3]) * v29_data);
              v342_acc += ((v349_data[4]) * v35_data);
              v342_acc += ((v349_data[5]) * v41_data);
              v342_acc += ((v349_data[6]) * v47_data);
              v342_acc += ((v349_data[7]) * v53_data);
              v342_acc += ((v349_data[8]) * v59_data);
              v342_acc += ((v349_data[9]) * v65_data);
              v342_acc += ((v349_data[10]) * v71_data);
              v342_acc += ((v349_data[11]) * v77_data);
              v342_acc += ((v349_data[12]) * v83_data);
              v342_acc += ((v349_data[13]) * v89_data);
              v342_acc += ((v349_data[14]) * v95_data);
              v342_acc += ((v349_data[15]) * v101_data);
              v342_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v382_acc{};
              tensorforge::intel_esimd::simd<float, 16> v389_data;
              v389_data.copy_from(s0 + ((112_i32 ^ ((112_i32 >> 4) & 15))));
              v382_acc += ((v389_data[0]) * v11_data);
              v382_acc += ((v389_data[1]) * v17_data);
              v382_acc += ((v389_data[2]) * v23_data);
              v382_acc += ((v389_data[3]) * v29_data);
              v382_acc += ((v389_data[4]) * v35_data);
              v382_acc += ((v389_data[5]) * v41_data);
              v382_acc += ((v389_data[6]) * v47_data);
              v382_acc += ((v389_data[7]) * v53_data);
              v382_acc += ((v389_data[8]) * v59_data);
              v382_acc += ((v389_data[9]) * v65_data);
              v382_acc += ((v389_data[10]) * v71_data);
              v382_acc += ((v389_data[11]) * v77_data);
              v382_acc += ((v389_data[12]) * v83_data);
              v382_acc += ((v389_data[13]) * v89_data);
              v382_acc += ((v389_data[14]) * v95_data);
              v382_acc += ((v389_data[15]) * v101_data);
              v382_acc.copy_to(ir0 + (112));
              #pragma unroll
              for (int32_t v422_n1 = 0; v422_n1 < 8; ++v422_n1) {
                int32_t v423_a = v422_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v425_data;
                v425_data.copy_from(ir0 + (v423_a));
                v425_data.copy_to(r0 + (v423_a));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v428_i1 = 0; v428_i1 < 8; ++v428_i1) {
                tensorforge::intel_esimd::simd<float, 12> v431_data;
                v431_data.copy_from(r0 + ((v428_i1 * 16)));
                v431_data.copy_to(glb_m0 + ((v428_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

