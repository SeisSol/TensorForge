// === base name ===
kernel_769f1b7f89

// === header ===
void launcher_kernel_769f1b7f89(float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_769f1b7f89(float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_769f1b7f89(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_769f1b7f89(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (4352, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×16(16×16) {0..16}×{0..16} pointer_based
        // m1 16×16(16×16) {0..16}×{0..16} pointer_based
        // m2 16×16(16×16) {0..16}×{0..16} pointer_based
        // m0 16×16(16×16) {0..16}×{0..16} pointer_based({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} pointer_based({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} pointer_based({0..16}×{0..16})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[272 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[256];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v6_i0 = 0; v6_i0 < 1; ++v6_i0) {
                int32_t v8_lead = v6_i0 * 16;
                #pragma unroll
                for (int32_t v7_i1 = 0; v7_i1 < 16; ++v7_i1) {
                  int32_t v11_a = v8_lead + (v7_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v12_data;
                  v12_data.copy_from(glb_m1 + (v11_a));
                  v12_data.copy_to(r0 + (v11_a));
                }
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 128] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 128];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 192] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 192];
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[256]{};
              // r1 = +(r0 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir1[256]{};
              tensorforge::intel_esimd::simd<float, 16> v19_data;
              v19_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v20_data;
              v20_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v21_data;
              v21_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v22_data;
              v22_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v24_data;
              v24_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v35_acc{};
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(s0 + ((0_i32 ^ ((0_i32 >> 5) & 31))));
              v35_acc += ((v42_data[0]) * v19_data);
              v35_acc += ((v42_data[1]) * v20_data);
              v35_acc += ((v42_data[2]) * v21_data);
              v35_acc += ((v42_data[3]) * v22_data);
              v35_acc += ((v42_data[4]) * v23_data);
              v35_acc += ((v42_data[5]) * v24_data);
              v35_acc += ((v42_data[6]) * v25_data);
              v35_acc += ((v42_data[7]) * v26_data);
              v35_acc += ((v42_data[8]) * v27_data);
              v35_acc += ((v42_data[9]) * v28_data);
              v35_acc += ((v42_data[10]) * v29_data);
              v35_acc += ((v42_data[11]) * v30_data);
              v35_acc += ((v42_data[12]) * v31_data);
              v35_acc += ((v42_data[13]) * v32_data);
              v35_acc += ((v42_data[14]) * v33_data);
              v35_acc += ((v42_data[15]) * v34_data);
              v35_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v75_acc{};
              tensorforge::intel_esimd::simd<float, 16> v82_data;
              v82_data.copy_from(s0 + ((16_i32 ^ ((16_i32 >> 5) & 31))));
              v75_acc += ((v82_data[0]) * v19_data);
              v75_acc += ((v82_data[1]) * v20_data);
              v75_acc += ((v82_data[2]) * v21_data);
              v75_acc += ((v82_data[3]) * v22_data);
              v75_acc += ((v82_data[4]) * v23_data);
              v75_acc += ((v82_data[5]) * v24_data);
              v75_acc += ((v82_data[6]) * v25_data);
              v75_acc += ((v82_data[7]) * v26_data);
              v75_acc += ((v82_data[8]) * v27_data);
              v75_acc += ((v82_data[9]) * v28_data);
              v75_acc += ((v82_data[10]) * v29_data);
              v75_acc += ((v82_data[11]) * v30_data);
              v75_acc += ((v82_data[12]) * v31_data);
              v75_acc += ((v82_data[13]) * v32_data);
              v75_acc += ((v82_data[14]) * v33_data);
              v75_acc += ((v82_data[15]) * v34_data);
              v75_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v115_acc{};
              tensorforge::intel_esimd::simd<float, 16> v122_data;
              v122_data.copy_from(s0 + ((32_i32 ^ ((32_i32 >> 5) & 31))));
              v115_acc += ((v122_data[0]) * v19_data);
              v115_acc += ((v122_data[1]) * v20_data);
              v115_acc += ((v122_data[2]) * v21_data);
              v115_acc += ((v122_data[3]) * v22_data);
              v115_acc += ((v122_data[4]) * v23_data);
              v115_acc += ((v122_data[5]) * v24_data);
              v115_acc += ((v122_data[6]) * v25_data);
              v115_acc += ((v122_data[7]) * v26_data);
              v115_acc += ((v122_data[8]) * v27_data);
              v115_acc += ((v122_data[9]) * v28_data);
              v115_acc += ((v122_data[10]) * v29_data);
              v115_acc += ((v122_data[11]) * v30_data);
              v115_acc += ((v122_data[12]) * v31_data);
              v115_acc += ((v122_data[13]) * v32_data);
              v115_acc += ((v122_data[14]) * v33_data);
              v115_acc += ((v122_data[15]) * v34_data);
              v115_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v155_acc{};
              tensorforge::intel_esimd::simd<float, 16> v162_data;
              v162_data.copy_from(s0 + ((48_i32 ^ ((48_i32 >> 5) & 31))));
              v155_acc += ((v162_data[0]) * v19_data);
              v155_acc += ((v162_data[1]) * v20_data);
              v155_acc += ((v162_data[2]) * v21_data);
              v155_acc += ((v162_data[3]) * v22_data);
              v155_acc += ((v162_data[4]) * v23_data);
              v155_acc += ((v162_data[5]) * v24_data);
              v155_acc += ((v162_data[6]) * v25_data);
              v155_acc += ((v162_data[7]) * v26_data);
              v155_acc += ((v162_data[8]) * v27_data);
              v155_acc += ((v162_data[9]) * v28_data);
              v155_acc += ((v162_data[10]) * v29_data);
              v155_acc += ((v162_data[11]) * v30_data);
              v155_acc += ((v162_data[12]) * v31_data);
              v155_acc += ((v162_data[13]) * v32_data);
              v155_acc += ((v162_data[14]) * v33_data);
              v155_acc += ((v162_data[15]) * v34_data);
              v155_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v195_acc{};
              tensorforge::intel_esimd::simd<float, 16> v202_data;
              v202_data.copy_from(s0 + ((64_i32 ^ ((64_i32 >> 5) & 31))));
              v195_acc += ((v202_data[0]) * v19_data);
              v195_acc += ((v202_data[1]) * v20_data);
              v195_acc += ((v202_data[2]) * v21_data);
              v195_acc += ((v202_data[3]) * v22_data);
              v195_acc += ((v202_data[4]) * v23_data);
              v195_acc += ((v202_data[5]) * v24_data);
              v195_acc += ((v202_data[6]) * v25_data);
              v195_acc += ((v202_data[7]) * v26_data);
              v195_acc += ((v202_data[8]) * v27_data);
              v195_acc += ((v202_data[9]) * v28_data);
              v195_acc += ((v202_data[10]) * v29_data);
              v195_acc += ((v202_data[11]) * v30_data);
              v195_acc += ((v202_data[12]) * v31_data);
              v195_acc += ((v202_data[13]) * v32_data);
              v195_acc += ((v202_data[14]) * v33_data);
              v195_acc += ((v202_data[15]) * v34_data);
              v195_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v235_acc{};
              tensorforge::intel_esimd::simd<float, 16> v242_data;
              v242_data.copy_from(s0 + ((80_i32 ^ ((80_i32 >> 5) & 31))));
              v235_acc += ((v242_data[0]) * v19_data);
              v235_acc += ((v242_data[1]) * v20_data);
              v235_acc += ((v242_data[2]) * v21_data);
              v235_acc += ((v242_data[3]) * v22_data);
              v235_acc += ((v242_data[4]) * v23_data);
              v235_acc += ((v242_data[5]) * v24_data);
              v235_acc += ((v242_data[6]) * v25_data);
              v235_acc += ((v242_data[7]) * v26_data);
              v235_acc += ((v242_data[8]) * v27_data);
              v235_acc += ((v242_data[9]) * v28_data);
              v235_acc += ((v242_data[10]) * v29_data);
              v235_acc += ((v242_data[11]) * v30_data);
              v235_acc += ((v242_data[12]) * v31_data);
              v235_acc += ((v242_data[13]) * v32_data);
              v235_acc += ((v242_data[14]) * v33_data);
              v235_acc += ((v242_data[15]) * v34_data);
              v235_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v275_acc{};
              tensorforge::intel_esimd::simd<float, 16> v282_data;
              v282_data.copy_from(s0 + ((96_i32 ^ ((96_i32 >> 5) & 31))));
              v275_acc += ((v282_data[0]) * v19_data);
              v275_acc += ((v282_data[1]) * v20_data);
              v275_acc += ((v282_data[2]) * v21_data);
              v275_acc += ((v282_data[3]) * v22_data);
              v275_acc += ((v282_data[4]) * v23_data);
              v275_acc += ((v282_data[5]) * v24_data);
              v275_acc += ((v282_data[6]) * v25_data);
              v275_acc += ((v282_data[7]) * v26_data);
              v275_acc += ((v282_data[8]) * v27_data);
              v275_acc += ((v282_data[9]) * v28_data);
              v275_acc += ((v282_data[10]) * v29_data);
              v275_acc += ((v282_data[11]) * v30_data);
              v275_acc += ((v282_data[12]) * v31_data);
              v275_acc += ((v282_data[13]) * v32_data);
              v275_acc += ((v282_data[14]) * v33_data);
              v275_acc += ((v282_data[15]) * v34_data);
              v275_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v315_acc{};
              tensorforge::intel_esimd::simd<float, 16> v322_data;
              v322_data.copy_from(s0 + ((112_i32 ^ ((112_i32 >> 5) & 31))));
              v315_acc += ((v322_data[0]) * v19_data);
              v315_acc += ((v322_data[1]) * v20_data);
              v315_acc += ((v322_data[2]) * v21_data);
              v315_acc += ((v322_data[3]) * v22_data);
              v315_acc += ((v322_data[4]) * v23_data);
              v315_acc += ((v322_data[5]) * v24_data);
              v315_acc += ((v322_data[6]) * v25_data);
              v315_acc += ((v322_data[7]) * v26_data);
              v315_acc += ((v322_data[8]) * v27_data);
              v315_acc += ((v322_data[9]) * v28_data);
              v315_acc += ((v322_data[10]) * v29_data);
              v315_acc += ((v322_data[11]) * v30_data);
              v315_acc += ((v322_data[12]) * v31_data);
              v315_acc += ((v322_data[13]) * v32_data);
              v315_acc += ((v322_data[14]) * v33_data);
              v315_acc += ((v322_data[15]) * v34_data);
              v315_acc.copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 16> v355_acc{};
              tensorforge::intel_esimd::simd<float, 16> v362_data;
              v362_data.copy_from(s0 + ((128_i32 ^ ((128_i32 >> 5) & 31))));
              v355_acc += ((v362_data[0]) * v19_data);
              v355_acc += ((v362_data[1]) * v20_data);
              v355_acc += ((v362_data[2]) * v21_data);
              v355_acc += ((v362_data[3]) * v22_data);
              v355_acc += ((v362_data[4]) * v23_data);
              v355_acc += ((v362_data[5]) * v24_data);
              v355_acc += ((v362_data[6]) * v25_data);
              v355_acc += ((v362_data[7]) * v26_data);
              v355_acc += ((v362_data[8]) * v27_data);
              v355_acc += ((v362_data[9]) * v28_data);
              v355_acc += ((v362_data[10]) * v29_data);
              v355_acc += ((v362_data[11]) * v30_data);
              v355_acc += ((v362_data[12]) * v31_data);
              v355_acc += ((v362_data[13]) * v32_data);
              v355_acc += ((v362_data[14]) * v33_data);
              v355_acc += ((v362_data[15]) * v34_data);
              v355_acc.copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 16> v395_acc{};
              tensorforge::intel_esimd::simd<float, 16> v402_data;
              v402_data.copy_from(s0 + ((144_i32 ^ ((144_i32 >> 5) & 31))));
              v395_acc += ((v402_data[0]) * v19_data);
              v395_acc += ((v402_data[1]) * v20_data);
              v395_acc += ((v402_data[2]) * v21_data);
              v395_acc += ((v402_data[3]) * v22_data);
              v395_acc += ((v402_data[4]) * v23_data);
              v395_acc += ((v402_data[5]) * v24_data);
              v395_acc += ((v402_data[6]) * v25_data);
              v395_acc += ((v402_data[7]) * v26_data);
              v395_acc += ((v402_data[8]) * v27_data);
              v395_acc += ((v402_data[9]) * v28_data);
              v395_acc += ((v402_data[10]) * v29_data);
              v395_acc += ((v402_data[11]) * v30_data);
              v395_acc += ((v402_data[12]) * v31_data);
              v395_acc += ((v402_data[13]) * v32_data);
              v395_acc += ((v402_data[14]) * v33_data);
              v395_acc += ((v402_data[15]) * v34_data);
              v395_acc.copy_to(ir1 + (144));
              tensorforge::intel_esimd::simd<float, 16> v435_acc{};
              tensorforge::intel_esimd::simd<float, 16> v442_data;
              v442_data.copy_from(s0 + ((160_i32 ^ ((160_i32 >> 5) & 31))));
              v435_acc += ((v442_data[0]) * v19_data);
              v435_acc += ((v442_data[1]) * v20_data);
              v435_acc += ((v442_data[2]) * v21_data);
              v435_acc += ((v442_data[3]) * v22_data);
              v435_acc += ((v442_data[4]) * v23_data);
              v435_acc += ((v442_data[5]) * v24_data);
              v435_acc += ((v442_data[6]) * v25_data);
              v435_acc += ((v442_data[7]) * v26_data);
              v435_acc += ((v442_data[8]) * v27_data);
              v435_acc += ((v442_data[9]) * v28_data);
              v435_acc += ((v442_data[10]) * v29_data);
              v435_acc += ((v442_data[11]) * v30_data);
              v435_acc += ((v442_data[12]) * v31_data);
              v435_acc += ((v442_data[13]) * v32_data);
              v435_acc += ((v442_data[14]) * v33_data);
              v435_acc += ((v442_data[15]) * v34_data);
              v435_acc.copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 16> v475_acc{};
              tensorforge::intel_esimd::simd<float, 16> v482_data;
              v482_data.copy_from(s0 + ((176_i32 ^ ((176_i32 >> 5) & 31))));
              v475_acc += ((v482_data[0]) * v19_data);
              v475_acc += ((v482_data[1]) * v20_data);
              v475_acc += ((v482_data[2]) * v21_data);
              v475_acc += ((v482_data[3]) * v22_data);
              v475_acc += ((v482_data[4]) * v23_data);
              v475_acc += ((v482_data[5]) * v24_data);
              v475_acc += ((v482_data[6]) * v25_data);
              v475_acc += ((v482_data[7]) * v26_data);
              v475_acc += ((v482_data[8]) * v27_data);
              v475_acc += ((v482_data[9]) * v28_data);
              v475_acc += ((v482_data[10]) * v29_data);
              v475_acc += ((v482_data[11]) * v30_data);
              v475_acc += ((v482_data[12]) * v31_data);
              v475_acc += ((v482_data[13]) * v32_data);
              v475_acc += ((v482_data[14]) * v33_data);
              v475_acc += ((v482_data[15]) * v34_data);
              v475_acc.copy_to(ir1 + (176));
              tensorforge::intel_esimd::simd<float, 16> v515_acc{};
              tensorforge::intel_esimd::simd<float, 16> v522_data;
              v522_data.copy_from(s0 + ((192_i32 ^ ((192_i32 >> 5) & 31))));
              v515_acc += ((v522_data[0]) * v19_data);
              v515_acc += ((v522_data[1]) * v20_data);
              v515_acc += ((v522_data[2]) * v21_data);
              v515_acc += ((v522_data[3]) * v22_data);
              v515_acc += ((v522_data[4]) * v23_data);
              v515_acc += ((v522_data[5]) * v24_data);
              v515_acc += ((v522_data[6]) * v25_data);
              v515_acc += ((v522_data[7]) * v26_data);
              v515_acc += ((v522_data[8]) * v27_data);
              v515_acc += ((v522_data[9]) * v28_data);
              v515_acc += ((v522_data[10]) * v29_data);
              v515_acc += ((v522_data[11]) * v30_data);
              v515_acc += ((v522_data[12]) * v31_data);
              v515_acc += ((v522_data[13]) * v32_data);
              v515_acc += ((v522_data[14]) * v33_data);
              v515_acc += ((v522_data[15]) * v34_data);
              v515_acc.copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 16> v555_acc{};
              tensorforge::intel_esimd::simd<float, 16> v562_data;
              v562_data.copy_from(s0 + ((208_i32 ^ ((208_i32 >> 5) & 31))));
              v555_acc += ((v562_data[0]) * v19_data);
              v555_acc += ((v562_data[1]) * v20_data);
              v555_acc += ((v562_data[2]) * v21_data);
              v555_acc += ((v562_data[3]) * v22_data);
              v555_acc += ((v562_data[4]) * v23_data);
              v555_acc += ((v562_data[5]) * v24_data);
              v555_acc += ((v562_data[6]) * v25_data);
              v555_acc += ((v562_data[7]) * v26_data);
              v555_acc += ((v562_data[8]) * v27_data);
              v555_acc += ((v562_data[9]) * v28_data);
              v555_acc += ((v562_data[10]) * v29_data);
              v555_acc += ((v562_data[11]) * v30_data);
              v555_acc += ((v562_data[12]) * v31_data);
              v555_acc += ((v562_data[13]) * v32_data);
              v555_acc += ((v562_data[14]) * v33_data);
              v555_acc += ((v562_data[15]) * v34_data);
              v555_acc.copy_to(ir1 + (208));
              tensorforge::intel_esimd::simd<float, 16> v595_acc{};
              tensorforge::intel_esimd::simd<float, 16> v602_data;
              v602_data.copy_from(s0 + ((224_i32 ^ ((224_i32 >> 5) & 31))));
              v595_acc += ((v602_data[0]) * v19_data);
              v595_acc += ((v602_data[1]) * v20_data);
              v595_acc += ((v602_data[2]) * v21_data);
              v595_acc += ((v602_data[3]) * v22_data);
              v595_acc += ((v602_data[4]) * v23_data);
              v595_acc += ((v602_data[5]) * v24_data);
              v595_acc += ((v602_data[6]) * v25_data);
              v595_acc += ((v602_data[7]) * v26_data);
              v595_acc += ((v602_data[8]) * v27_data);
              v595_acc += ((v602_data[9]) * v28_data);
              v595_acc += ((v602_data[10]) * v29_data);
              v595_acc += ((v602_data[11]) * v30_data);
              v595_acc += ((v602_data[12]) * v31_data);
              v595_acc += ((v602_data[13]) * v32_data);
              v595_acc += ((v602_data[14]) * v33_data);
              v595_acc += ((v602_data[15]) * v34_data);
              v595_acc.copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 16> v635_acc{};
              tensorforge::intel_esimd::simd<float, 16> v642_data;
              v642_data.copy_from(s0 + ((240_i32 ^ ((240_i32 >> 5) & 31))));
              v635_acc += ((v642_data[0]) * v19_data);
              v635_acc += ((v642_data[1]) * v20_data);
              v635_acc += ((v642_data[2]) * v21_data);
              v635_acc += ((v642_data[3]) * v22_data);
              v635_acc += ((v642_data[4]) * v23_data);
              v635_acc += ((v642_data[5]) * v24_data);
              v635_acc += ((v642_data[6]) * v25_data);
              v635_acc += ((v642_data[7]) * v26_data);
              v635_acc += ((v642_data[8]) * v27_data);
              v635_acc += ((v642_data[9]) * v28_data);
              v635_acc += ((v642_data[10]) * v29_data);
              v635_acc += ((v642_data[11]) * v30_data);
              v635_acc += ((v642_data[12]) * v31_data);
              v635_acc += ((v642_data[13]) * v32_data);
              v635_acc += ((v642_data[14]) * v33_data);
              v635_acc += ((v642_data[15]) * v34_data);
              v635_acc.copy_to(ir1 + (240));
              #pragma unroll
              for (int32_t v675_n0 = 0; v675_n0 < 1; ++v675_n0) {
                int32_t v677_a = v675_n0 * 16;
                #pragma unroll
                for (int32_t v676_n1 = 0; v676_n1 < 16; ++v676_n1) {
                  int32_t v679_a = v677_a + (v676_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v680_data;
                  v680_data.copy_from(ir1 + (v679_a));
                  v680_data.copy_to(r1 + (v679_a));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v684_i0 = 0; v684_i0 < 1; ++v684_i0) {
                int32_t v686_a = v684_i0 * 16;
                #pragma unroll
                for (int32_t v685_i1 = 0; v685_i1 < 16; ++v685_i1) {
                  int32_t v688_a = v686_a + (v685_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v689_data;
                  v689_data.copy_from(r1 + (v688_a));
                  v689_data.copy_to(glb_m0 + (v688_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

