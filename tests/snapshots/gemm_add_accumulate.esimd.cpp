// === base name ===
kernel_5e7da3148f

// === header ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5e7da3148f(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5e7da3148f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
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
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v6_i1 = 0; v6_i1 < 16; ++v6_i1) {
                tensorforge::intel_esimd::simd<float, 12> v11_data;
                v11_data.copy_from(glb_m1 + ((v6_i1 * 12)));
                v11_data.copy_to(r0 + ((v6_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(r0 = load{g>r}(glb_m1););
              float r1[128]{};
              // r1 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v16_i1 = 0; v16_i1 < 8; ++v16_i1) {
                tensorforge::intel_esimd::simd<float, 12> v21_data;
                v21_data.copy_from(glb_m0 + ((v16_i1 * 12)));
                v21_data.copy_to(r1 + ((v16_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              // wait(r1 = load{g>r}(glb_m0););
              float r2[128]{};
              // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir2[128]{};
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v42_acc{};
              tensorforge::intel_esimd::simd<float, 16> v49_data;
              v49_data.copy_from(s0 + ((0_i32 ^ ((0_i32 >> 5) & 31))));
              v42_acc += ((v49_data[0]) * v26_data);
              v42_acc += ((v49_data[1]) * v27_data);
              v42_acc += ((v49_data[2]) * v28_data);
              v42_acc += ((v49_data[3]) * v29_data);
              v42_acc += ((v49_data[4]) * v30_data);
              v42_acc += ((v49_data[5]) * v31_data);
              v42_acc += ((v49_data[6]) * v32_data);
              v42_acc += ((v49_data[7]) * v33_data);
              v42_acc += ((v49_data[8]) * v34_data);
              v42_acc += ((v49_data[9]) * v35_data);
              v42_acc += ((v49_data[10]) * v36_data);
              v42_acc += ((v49_data[11]) * v37_data);
              v42_acc += ((v49_data[12]) * v38_data);
              v42_acc += ((v49_data[13]) * v39_data);
              v42_acc += ((v49_data[14]) * v40_data);
              v42_acc += ((v49_data[15]) * v41_data);
              v42_acc.copy_to(ir2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v82_acc{};
              tensorforge::intel_esimd::simd<float, 16> v89_data;
              v89_data.copy_from(s0 + ((16_i32 ^ ((16_i32 >> 5) & 31))));
              v82_acc += ((v89_data[0]) * v26_data);
              v82_acc += ((v89_data[1]) * v27_data);
              v82_acc += ((v89_data[2]) * v28_data);
              v82_acc += ((v89_data[3]) * v29_data);
              v82_acc += ((v89_data[4]) * v30_data);
              v82_acc += ((v89_data[5]) * v31_data);
              v82_acc += ((v89_data[6]) * v32_data);
              v82_acc += ((v89_data[7]) * v33_data);
              v82_acc += ((v89_data[8]) * v34_data);
              v82_acc += ((v89_data[9]) * v35_data);
              v82_acc += ((v89_data[10]) * v36_data);
              v82_acc += ((v89_data[11]) * v37_data);
              v82_acc += ((v89_data[12]) * v38_data);
              v82_acc += ((v89_data[13]) * v39_data);
              v82_acc += ((v89_data[14]) * v40_data);
              v82_acc += ((v89_data[15]) * v41_data);
              v82_acc.copy_to(ir2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v122_acc{};
              tensorforge::intel_esimd::simd<float, 16> v129_data;
              v129_data.copy_from(s0 + ((32_i32 ^ ((32_i32 >> 5) & 31))));
              v122_acc += ((v129_data[0]) * v26_data);
              v122_acc += ((v129_data[1]) * v27_data);
              v122_acc += ((v129_data[2]) * v28_data);
              v122_acc += ((v129_data[3]) * v29_data);
              v122_acc += ((v129_data[4]) * v30_data);
              v122_acc += ((v129_data[5]) * v31_data);
              v122_acc += ((v129_data[6]) * v32_data);
              v122_acc += ((v129_data[7]) * v33_data);
              v122_acc += ((v129_data[8]) * v34_data);
              v122_acc += ((v129_data[9]) * v35_data);
              v122_acc += ((v129_data[10]) * v36_data);
              v122_acc += ((v129_data[11]) * v37_data);
              v122_acc += ((v129_data[12]) * v38_data);
              v122_acc += ((v129_data[13]) * v39_data);
              v122_acc += ((v129_data[14]) * v40_data);
              v122_acc += ((v129_data[15]) * v41_data);
              v122_acc.copy_to(ir2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v162_acc{};
              tensorforge::intel_esimd::simd<float, 16> v169_data;
              v169_data.copy_from(s0 + ((48_i32 ^ ((48_i32 >> 5) & 31))));
              v162_acc += ((v169_data[0]) * v26_data);
              v162_acc += ((v169_data[1]) * v27_data);
              v162_acc += ((v169_data[2]) * v28_data);
              v162_acc += ((v169_data[3]) * v29_data);
              v162_acc += ((v169_data[4]) * v30_data);
              v162_acc += ((v169_data[5]) * v31_data);
              v162_acc += ((v169_data[6]) * v32_data);
              v162_acc += ((v169_data[7]) * v33_data);
              v162_acc += ((v169_data[8]) * v34_data);
              v162_acc += ((v169_data[9]) * v35_data);
              v162_acc += ((v169_data[10]) * v36_data);
              v162_acc += ((v169_data[11]) * v37_data);
              v162_acc += ((v169_data[12]) * v38_data);
              v162_acc += ((v169_data[13]) * v39_data);
              v162_acc += ((v169_data[14]) * v40_data);
              v162_acc += ((v169_data[15]) * v41_data);
              v162_acc.copy_to(ir2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v202_acc{};
              tensorforge::intel_esimd::simd<float, 16> v209_data;
              v209_data.copy_from(s0 + ((64_i32 ^ ((64_i32 >> 5) & 31))));
              v202_acc += ((v209_data[0]) * v26_data);
              v202_acc += ((v209_data[1]) * v27_data);
              v202_acc += ((v209_data[2]) * v28_data);
              v202_acc += ((v209_data[3]) * v29_data);
              v202_acc += ((v209_data[4]) * v30_data);
              v202_acc += ((v209_data[5]) * v31_data);
              v202_acc += ((v209_data[6]) * v32_data);
              v202_acc += ((v209_data[7]) * v33_data);
              v202_acc += ((v209_data[8]) * v34_data);
              v202_acc += ((v209_data[9]) * v35_data);
              v202_acc += ((v209_data[10]) * v36_data);
              v202_acc += ((v209_data[11]) * v37_data);
              v202_acc += ((v209_data[12]) * v38_data);
              v202_acc += ((v209_data[13]) * v39_data);
              v202_acc += ((v209_data[14]) * v40_data);
              v202_acc += ((v209_data[15]) * v41_data);
              v202_acc.copy_to(ir2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v242_acc{};
              tensorforge::intel_esimd::simd<float, 16> v249_data;
              v249_data.copy_from(s0 + ((80_i32 ^ ((80_i32 >> 5) & 31))));
              v242_acc += ((v249_data[0]) * v26_data);
              v242_acc += ((v249_data[1]) * v27_data);
              v242_acc += ((v249_data[2]) * v28_data);
              v242_acc += ((v249_data[3]) * v29_data);
              v242_acc += ((v249_data[4]) * v30_data);
              v242_acc += ((v249_data[5]) * v31_data);
              v242_acc += ((v249_data[6]) * v32_data);
              v242_acc += ((v249_data[7]) * v33_data);
              v242_acc += ((v249_data[8]) * v34_data);
              v242_acc += ((v249_data[9]) * v35_data);
              v242_acc += ((v249_data[10]) * v36_data);
              v242_acc += ((v249_data[11]) * v37_data);
              v242_acc += ((v249_data[12]) * v38_data);
              v242_acc += ((v249_data[13]) * v39_data);
              v242_acc += ((v249_data[14]) * v40_data);
              v242_acc += ((v249_data[15]) * v41_data);
              v242_acc.copy_to(ir2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v282_acc{};
              tensorforge::intel_esimd::simd<float, 16> v289_data;
              v289_data.copy_from(s0 + ((96_i32 ^ ((96_i32 >> 5) & 31))));
              v282_acc += ((v289_data[0]) * v26_data);
              v282_acc += ((v289_data[1]) * v27_data);
              v282_acc += ((v289_data[2]) * v28_data);
              v282_acc += ((v289_data[3]) * v29_data);
              v282_acc += ((v289_data[4]) * v30_data);
              v282_acc += ((v289_data[5]) * v31_data);
              v282_acc += ((v289_data[6]) * v32_data);
              v282_acc += ((v289_data[7]) * v33_data);
              v282_acc += ((v289_data[8]) * v34_data);
              v282_acc += ((v289_data[9]) * v35_data);
              v282_acc += ((v289_data[10]) * v36_data);
              v282_acc += ((v289_data[11]) * v37_data);
              v282_acc += ((v289_data[12]) * v38_data);
              v282_acc += ((v289_data[13]) * v39_data);
              v282_acc += ((v289_data[14]) * v40_data);
              v282_acc += ((v289_data[15]) * v41_data);
              v282_acc.copy_to(ir2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v322_acc{};
              tensorforge::intel_esimd::simd<float, 16> v329_data;
              v329_data.copy_from(s0 + ((112_i32 ^ ((112_i32 >> 5) & 31))));
              v322_acc += ((v329_data[0]) * v26_data);
              v322_acc += ((v329_data[1]) * v27_data);
              v322_acc += ((v329_data[2]) * v28_data);
              v322_acc += ((v329_data[3]) * v29_data);
              v322_acc += ((v329_data[4]) * v30_data);
              v322_acc += ((v329_data[5]) * v31_data);
              v322_acc += ((v329_data[6]) * v32_data);
              v322_acc += ((v329_data[7]) * v33_data);
              v322_acc += ((v329_data[8]) * v34_data);
              v322_acc += ((v329_data[9]) * v35_data);
              v322_acc += ((v329_data[10]) * v36_data);
              v322_acc += ((v329_data[11]) * v37_data);
              v322_acc += ((v329_data[12]) * v38_data);
              v322_acc += ((v329_data[13]) * v39_data);
              v322_acc += ((v329_data[14]) * v40_data);
              v322_acc += ((v329_data[15]) * v41_data);
              v322_acc.copy_to(ir2 + (112));
              #pragma unroll
              for (int32_t v362_n1 = 0; v362_n1 < 8; ++v362_n1) {
                int32_t v363_a = v362_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v365_data;
                v365_data.copy_from(ir2 + (v363_a));
                tensorforge::intel_esimd::simd<float, 12> v368_data;
                v368_data.copy_from(r1 + (v363_a));
                (v368_data + v365_data).copy_to(r2 + (v363_a));
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v372_i1 = 0; v372_i1 < 8; ++v372_i1) {
                tensorforge::intel_esimd::simd<float, 12> v375_data;
                v375_data.copy_from(r2 + ((v372_i1 * 16)));
                v375_data.copy_to(glb_m0 + ((v372_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

