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
              tensorforge::intel_esimd::simd<float, 16> v106_data;
              v106_data.copy_from(s0 + (0_i32));
              v102_acc += ((v106_data[0]) * v11_data);
              v102_acc += ((v106_data[1]) * v17_data);
              v102_acc += ((v106_data[2]) * v23_data);
              v102_acc += ((v106_data[3]) * v29_data);
              v102_acc += ((v106_data[4]) * v35_data);
              v102_acc += ((v106_data[5]) * v41_data);
              v102_acc += ((v106_data[6]) * v47_data);
              v102_acc += ((v106_data[7]) * v53_data);
              v102_acc += ((v106_data[8]) * v59_data);
              v102_acc += ((v106_data[9]) * v65_data);
              v102_acc += ((v106_data[10]) * v71_data);
              v102_acc += ((v106_data[11]) * v77_data);
              v102_acc += ((v106_data[12]) * v83_data);
              v102_acc += ((v106_data[13]) * v89_data);
              v102_acc += ((v106_data[14]) * v95_data);
              v102_acc += ((v106_data[15]) * v101_data);
              v102_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v139_acc{};
              tensorforge::intel_esimd::simd<float, 16> v143_data;
              v143_data.copy_from(s0 + (16_i32));
              v139_acc += ((v143_data[0]) * v11_data);
              v139_acc += ((v143_data[1]) * v17_data);
              v139_acc += ((v143_data[2]) * v23_data);
              v139_acc += ((v143_data[3]) * v29_data);
              v139_acc += ((v143_data[4]) * v35_data);
              v139_acc += ((v143_data[5]) * v41_data);
              v139_acc += ((v143_data[6]) * v47_data);
              v139_acc += ((v143_data[7]) * v53_data);
              v139_acc += ((v143_data[8]) * v59_data);
              v139_acc += ((v143_data[9]) * v65_data);
              v139_acc += ((v143_data[10]) * v71_data);
              v139_acc += ((v143_data[11]) * v77_data);
              v139_acc += ((v143_data[12]) * v83_data);
              v139_acc += ((v143_data[13]) * v89_data);
              v139_acc += ((v143_data[14]) * v95_data);
              v139_acc += ((v143_data[15]) * v101_data);
              v139_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v176_acc{};
              tensorforge::intel_esimd::simd<float, 16> v180_data;
              v180_data.copy_from(s0 + (32_i32));
              v176_acc += ((v180_data[0]) * v11_data);
              v176_acc += ((v180_data[1]) * v17_data);
              v176_acc += ((v180_data[2]) * v23_data);
              v176_acc += ((v180_data[3]) * v29_data);
              v176_acc += ((v180_data[4]) * v35_data);
              v176_acc += ((v180_data[5]) * v41_data);
              v176_acc += ((v180_data[6]) * v47_data);
              v176_acc += ((v180_data[7]) * v53_data);
              v176_acc += ((v180_data[8]) * v59_data);
              v176_acc += ((v180_data[9]) * v65_data);
              v176_acc += ((v180_data[10]) * v71_data);
              v176_acc += ((v180_data[11]) * v77_data);
              v176_acc += ((v180_data[12]) * v83_data);
              v176_acc += ((v180_data[13]) * v89_data);
              v176_acc += ((v180_data[14]) * v95_data);
              v176_acc += ((v180_data[15]) * v101_data);
              v176_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v213_acc{};
              tensorforge::intel_esimd::simd<float, 16> v217_data;
              v217_data.copy_from(s0 + (48_i32));
              v213_acc += ((v217_data[0]) * v11_data);
              v213_acc += ((v217_data[1]) * v17_data);
              v213_acc += ((v217_data[2]) * v23_data);
              v213_acc += ((v217_data[3]) * v29_data);
              v213_acc += ((v217_data[4]) * v35_data);
              v213_acc += ((v217_data[5]) * v41_data);
              v213_acc += ((v217_data[6]) * v47_data);
              v213_acc += ((v217_data[7]) * v53_data);
              v213_acc += ((v217_data[8]) * v59_data);
              v213_acc += ((v217_data[9]) * v65_data);
              v213_acc += ((v217_data[10]) * v71_data);
              v213_acc += ((v217_data[11]) * v77_data);
              v213_acc += ((v217_data[12]) * v83_data);
              v213_acc += ((v217_data[13]) * v89_data);
              v213_acc += ((v217_data[14]) * v95_data);
              v213_acc += ((v217_data[15]) * v101_data);
              v213_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v250_acc{};
              tensorforge::intel_esimd::simd<float, 16> v254_data;
              v254_data.copy_from(s0 + (64_i32));
              v250_acc += ((v254_data[0]) * v11_data);
              v250_acc += ((v254_data[1]) * v17_data);
              v250_acc += ((v254_data[2]) * v23_data);
              v250_acc += ((v254_data[3]) * v29_data);
              v250_acc += ((v254_data[4]) * v35_data);
              v250_acc += ((v254_data[5]) * v41_data);
              v250_acc += ((v254_data[6]) * v47_data);
              v250_acc += ((v254_data[7]) * v53_data);
              v250_acc += ((v254_data[8]) * v59_data);
              v250_acc += ((v254_data[9]) * v65_data);
              v250_acc += ((v254_data[10]) * v71_data);
              v250_acc += ((v254_data[11]) * v77_data);
              v250_acc += ((v254_data[12]) * v83_data);
              v250_acc += ((v254_data[13]) * v89_data);
              v250_acc += ((v254_data[14]) * v95_data);
              v250_acc += ((v254_data[15]) * v101_data);
              v250_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v287_acc{};
              tensorforge::intel_esimd::simd<float, 16> v291_data;
              v291_data.copy_from(s0 + (80_i32));
              v287_acc += ((v291_data[0]) * v11_data);
              v287_acc += ((v291_data[1]) * v17_data);
              v287_acc += ((v291_data[2]) * v23_data);
              v287_acc += ((v291_data[3]) * v29_data);
              v287_acc += ((v291_data[4]) * v35_data);
              v287_acc += ((v291_data[5]) * v41_data);
              v287_acc += ((v291_data[6]) * v47_data);
              v287_acc += ((v291_data[7]) * v53_data);
              v287_acc += ((v291_data[8]) * v59_data);
              v287_acc += ((v291_data[9]) * v65_data);
              v287_acc += ((v291_data[10]) * v71_data);
              v287_acc += ((v291_data[11]) * v77_data);
              v287_acc += ((v291_data[12]) * v83_data);
              v287_acc += ((v291_data[13]) * v89_data);
              v287_acc += ((v291_data[14]) * v95_data);
              v287_acc += ((v291_data[15]) * v101_data);
              v287_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v324_acc{};
              tensorforge::intel_esimd::simd<float, 16> v328_data;
              v328_data.copy_from(s0 + (96_i32));
              v324_acc += ((v328_data[0]) * v11_data);
              v324_acc += ((v328_data[1]) * v17_data);
              v324_acc += ((v328_data[2]) * v23_data);
              v324_acc += ((v328_data[3]) * v29_data);
              v324_acc += ((v328_data[4]) * v35_data);
              v324_acc += ((v328_data[5]) * v41_data);
              v324_acc += ((v328_data[6]) * v47_data);
              v324_acc += ((v328_data[7]) * v53_data);
              v324_acc += ((v328_data[8]) * v59_data);
              v324_acc += ((v328_data[9]) * v65_data);
              v324_acc += ((v328_data[10]) * v71_data);
              v324_acc += ((v328_data[11]) * v77_data);
              v324_acc += ((v328_data[12]) * v83_data);
              v324_acc += ((v328_data[13]) * v89_data);
              v324_acc += ((v328_data[14]) * v95_data);
              v324_acc += ((v328_data[15]) * v101_data);
              v324_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v361_acc{};
              tensorforge::intel_esimd::simd<float, 16> v365_data;
              v365_data.copy_from(s0 + (112_i32));
              v361_acc += ((v365_data[0]) * v11_data);
              v361_acc += ((v365_data[1]) * v17_data);
              v361_acc += ((v365_data[2]) * v23_data);
              v361_acc += ((v365_data[3]) * v29_data);
              v361_acc += ((v365_data[4]) * v35_data);
              v361_acc += ((v365_data[5]) * v41_data);
              v361_acc += ((v365_data[6]) * v47_data);
              v361_acc += ((v365_data[7]) * v53_data);
              v361_acc += ((v365_data[8]) * v59_data);
              v361_acc += ((v365_data[9]) * v65_data);
              v361_acc += ((v365_data[10]) * v71_data);
              v361_acc += ((v365_data[11]) * v77_data);
              v361_acc += ((v365_data[12]) * v83_data);
              v361_acc += ((v365_data[13]) * v89_data);
              v361_acc += ((v365_data[14]) * v95_data);
              v361_acc += ((v365_data[15]) * v101_data);
              v361_acc.copy_to(ir0 + (112));
              #pragma unroll
              for (int32_t v398_n1 = 0; v398_n1 < 8; ++v398_n1) {
                int32_t v399_a = v398_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v401_data;
                v401_data.copy_from(ir0 + (v399_a));
                v401_data.copy_to(r0 + (v399_a));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v404_i1 = 0; v404_i1 < 8; ++v404_i1) {
                tensorforge::intel_esimd::simd<float, 12> v407_data;
                v407_data.copy_from(r0 + ((v404_i1 * 16)));
                v407_data.copy_to(glb_m0 + ((v404_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

