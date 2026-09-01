// === base name ===
kernel_f94e030d8c

// === header ===
void launcher_kernel_f94e030d8c(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f94e030d8c(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f94e030d8c(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f94e030d8c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (9472, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×16(12×16) {0..12}×{0..16} strided
        // m1 20×12(20×12) {0..20}×{0..12} strided
        // m2 20×16(20×16) {0..20}×{0..16} strided
        // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 20×12(20×12) {0..20}×{0..12} strided({0..20}×{0..12})[-1, 0]×m2 20×16(20×16) {0..20}×{0..16} strided({0..20}×{0..16})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[592 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[576];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 320 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[320];
              // s0 = load{g>s}(glb_m1[1, 0])
              #pragma unroll
              for (int32_t v6_i0 = 0; v6_i0 < 1; ++v6_i0) {
                int32_t v8_lead = v6_i0 * 16;
                #pragma unroll
                for (int32_t v7_i1 = 0; v7_i1 < 12; ++v7_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v12_data;
                  v12_data.copy_from(glb_m1 + ((v8_lead + (v7_i1 * 20))));
                  v12_data.copy_to(s0 + ((v8_lead + (v7_i1 * 21))));
                }
              }
              #pragma unroll
              for (int32_t v17_i1 = 0; v17_i1 < 12; ++v17_i1) {
                tensorforge::intel_esimd::simd<float, 4> v23_data;
                v23_data.copy_from(glb_m1 + ((16_i32 + (v17_i1 * 20))));
                v23_data.copy_to(s0 + ((16_i32 + (v17_i1 * 21))));
              }
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m2[0, 1])
              #pragma unroll
              for (int32_t i = 0; i < 20; i += 4) {
                tensorforge::intel_esimd::simd<float, 64> v30_ld;
                v30_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + i * 16));
                v30_ld.copy_to(s1 + (0 + 0 + 4 * item.get_local_id(0) + i * 16));
              }
              // wait(s0 = load{g>s}(glb_m1[1, 0]));
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              float r0[256]{};
              // r0 = +(s0 * s1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir0[256]{};
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(s0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(s0 + (1_i32));
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(s0 + (2_i32));
              tensorforge::intel_esimd::simd<float, 16> v52_data;
              v52_data.copy_from(s0 + (3_i32));
              tensorforge::intel_esimd::simd<float, 16> v57_data;
              v57_data.copy_from(s0 + (4_i32));
              tensorforge::intel_esimd::simd<float, 16> v62_data;
              v62_data.copy_from(s0 + (5_i32));
              tensorforge::intel_esimd::simd<float, 16> v67_data;
              v67_data.copy_from(s0 + (6_i32));
              tensorforge::intel_esimd::simd<float, 16> v72_data;
              v72_data.copy_from(s0 + (7_i32));
              tensorforge::intel_esimd::simd<float, 16> v77_data;
              v77_data.copy_from(s0 + (8_i32));
              tensorforge::intel_esimd::simd<float, 16> v82_data;
              v82_data.copy_from(s0 + (9_i32));
              tensorforge::intel_esimd::simd<float, 16> v87_data;
              v87_data.copy_from(s0 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v92_data;
              v92_data.copy_from(s0 + (11_i32));
              tensorforge::intel_esimd::simd<float, 16> v97_data;
              v97_data.copy_from(s0 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v102_data;
              v102_data.copy_from(s0 + (13_i32));
              tensorforge::intel_esimd::simd<float, 16> v107_data;
              v107_data.copy_from(s0 + (14_i32));
              tensorforge::intel_esimd::simd<float, 16> v112_data;
              v112_data.copy_from(s0 + (15_i32));
              tensorforge::intel_esimd::simd<float, 16> v117_data;
              v117_data.copy_from(s0 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v122_data;
              v122_data.copy_from(s0 + (17_i32));
              tensorforge::intel_esimd::simd<float, 16> v127_data;
              v127_data.copy_from(s0 + (18_i32));
              tensorforge::intel_esimd::simd<float, 16> v132_data;
              v132_data.copy_from(s0 + (19_i32));
              tensorforge::intel_esimd::simd<float, 16> v133_acc{};
              tensorforge::intel_esimd::simd<float, 16> v137_data;
              v137_data.copy_from(s1 + (0_i32));
              v133_acc += ((v137_data[0]) * v37_data);
              v133_acc += ((v137_data[1]) * v42_data);
              v133_acc += ((v137_data[2]) * v47_data);
              v133_acc += ((v137_data[3]) * v52_data);
              v133_acc += ((v137_data[4]) * v57_data);
              v133_acc += ((v137_data[5]) * v62_data);
              v133_acc += ((v137_data[6]) * v67_data);
              v133_acc += ((v137_data[7]) * v72_data);
              v133_acc += ((v137_data[8]) * v77_data);
              v133_acc += ((v137_data[9]) * v82_data);
              v133_acc += ((v137_data[10]) * v87_data);
              v133_acc += ((v137_data[11]) * v92_data);
              v133_acc += ((v137_data[12]) * v97_data);
              v133_acc += ((v137_data[13]) * v102_data);
              v133_acc += ((v137_data[14]) * v107_data);
              v133_acc += ((v137_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v173_data;
              v173_data.copy_from(s1 + (16_i32));
              v133_acc += ((v173_data[0]) * v117_data);
              v133_acc += ((v173_data[1]) * v122_data);
              v133_acc += ((v173_data[2]) * v127_data);
              v133_acc += ((v173_data[3]) * v132_data);
              v133_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v182_acc{};
              tensorforge::intel_esimd::simd<float, 16> v186_data;
              v186_data.copy_from(s1 + (20_i32));
              v182_acc += ((v186_data[0]) * v37_data);
              v182_acc += ((v186_data[1]) * v42_data);
              v182_acc += ((v186_data[2]) * v47_data);
              v182_acc += ((v186_data[3]) * v52_data);
              v182_acc += ((v186_data[4]) * v57_data);
              v182_acc += ((v186_data[5]) * v62_data);
              v182_acc += ((v186_data[6]) * v67_data);
              v182_acc += ((v186_data[7]) * v72_data);
              v182_acc += ((v186_data[8]) * v77_data);
              v182_acc += ((v186_data[9]) * v82_data);
              v182_acc += ((v186_data[10]) * v87_data);
              v182_acc += ((v186_data[11]) * v92_data);
              v182_acc += ((v186_data[12]) * v97_data);
              v182_acc += ((v186_data[13]) * v102_data);
              v182_acc += ((v186_data[14]) * v107_data);
              v182_acc += ((v186_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v222_data;
              v222_data.copy_from(s1 + (36_i32));
              v182_acc += ((v222_data[0]) * v117_data);
              v182_acc += ((v222_data[1]) * v122_data);
              v182_acc += ((v222_data[2]) * v127_data);
              v182_acc += ((v222_data[3]) * v132_data);
              v182_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v231_acc{};
              tensorforge::intel_esimd::simd<float, 16> v235_data;
              v235_data.copy_from(s1 + (40_i32));
              v231_acc += ((v235_data[0]) * v37_data);
              v231_acc += ((v235_data[1]) * v42_data);
              v231_acc += ((v235_data[2]) * v47_data);
              v231_acc += ((v235_data[3]) * v52_data);
              v231_acc += ((v235_data[4]) * v57_data);
              v231_acc += ((v235_data[5]) * v62_data);
              v231_acc += ((v235_data[6]) * v67_data);
              v231_acc += ((v235_data[7]) * v72_data);
              v231_acc += ((v235_data[8]) * v77_data);
              v231_acc += ((v235_data[9]) * v82_data);
              v231_acc += ((v235_data[10]) * v87_data);
              v231_acc += ((v235_data[11]) * v92_data);
              v231_acc += ((v235_data[12]) * v97_data);
              v231_acc += ((v235_data[13]) * v102_data);
              v231_acc += ((v235_data[14]) * v107_data);
              v231_acc += ((v235_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v271_data;
              v271_data.copy_from(s1 + (56_i32));
              v231_acc += ((v271_data[0]) * v117_data);
              v231_acc += ((v271_data[1]) * v122_data);
              v231_acc += ((v271_data[2]) * v127_data);
              v231_acc += ((v271_data[3]) * v132_data);
              v231_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v280_acc{};
              tensorforge::intel_esimd::simd<float, 16> v284_data;
              v284_data.copy_from(s1 + (60_i32));
              v280_acc += ((v284_data[0]) * v37_data);
              v280_acc += ((v284_data[1]) * v42_data);
              v280_acc += ((v284_data[2]) * v47_data);
              v280_acc += ((v284_data[3]) * v52_data);
              v280_acc += ((v284_data[4]) * v57_data);
              v280_acc += ((v284_data[5]) * v62_data);
              v280_acc += ((v284_data[6]) * v67_data);
              v280_acc += ((v284_data[7]) * v72_data);
              v280_acc += ((v284_data[8]) * v77_data);
              v280_acc += ((v284_data[9]) * v82_data);
              v280_acc += ((v284_data[10]) * v87_data);
              v280_acc += ((v284_data[11]) * v92_data);
              v280_acc += ((v284_data[12]) * v97_data);
              v280_acc += ((v284_data[13]) * v102_data);
              v280_acc += ((v284_data[14]) * v107_data);
              v280_acc += ((v284_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v320_data;
              v320_data.copy_from(s1 + (76_i32));
              v280_acc += ((v320_data[0]) * v117_data);
              v280_acc += ((v320_data[1]) * v122_data);
              v280_acc += ((v320_data[2]) * v127_data);
              v280_acc += ((v320_data[3]) * v132_data);
              v280_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v329_acc{};
              tensorforge::intel_esimd::simd<float, 16> v333_data;
              v333_data.copy_from(s1 + (80_i32));
              v329_acc += ((v333_data[0]) * v37_data);
              v329_acc += ((v333_data[1]) * v42_data);
              v329_acc += ((v333_data[2]) * v47_data);
              v329_acc += ((v333_data[3]) * v52_data);
              v329_acc += ((v333_data[4]) * v57_data);
              v329_acc += ((v333_data[5]) * v62_data);
              v329_acc += ((v333_data[6]) * v67_data);
              v329_acc += ((v333_data[7]) * v72_data);
              v329_acc += ((v333_data[8]) * v77_data);
              v329_acc += ((v333_data[9]) * v82_data);
              v329_acc += ((v333_data[10]) * v87_data);
              v329_acc += ((v333_data[11]) * v92_data);
              v329_acc += ((v333_data[12]) * v97_data);
              v329_acc += ((v333_data[13]) * v102_data);
              v329_acc += ((v333_data[14]) * v107_data);
              v329_acc += ((v333_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v369_data;
              v369_data.copy_from(s1 + (96_i32));
              v329_acc += ((v369_data[0]) * v117_data);
              v329_acc += ((v369_data[1]) * v122_data);
              v329_acc += ((v369_data[2]) * v127_data);
              v329_acc += ((v369_data[3]) * v132_data);
              v329_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v378_acc{};
              tensorforge::intel_esimd::simd<float, 16> v382_data;
              v382_data.copy_from(s1 + (100_i32));
              v378_acc += ((v382_data[0]) * v37_data);
              v378_acc += ((v382_data[1]) * v42_data);
              v378_acc += ((v382_data[2]) * v47_data);
              v378_acc += ((v382_data[3]) * v52_data);
              v378_acc += ((v382_data[4]) * v57_data);
              v378_acc += ((v382_data[5]) * v62_data);
              v378_acc += ((v382_data[6]) * v67_data);
              v378_acc += ((v382_data[7]) * v72_data);
              v378_acc += ((v382_data[8]) * v77_data);
              v378_acc += ((v382_data[9]) * v82_data);
              v378_acc += ((v382_data[10]) * v87_data);
              v378_acc += ((v382_data[11]) * v92_data);
              v378_acc += ((v382_data[12]) * v97_data);
              v378_acc += ((v382_data[13]) * v102_data);
              v378_acc += ((v382_data[14]) * v107_data);
              v378_acc += ((v382_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v418_data;
              v418_data.copy_from(s1 + (116_i32));
              v378_acc += ((v418_data[0]) * v117_data);
              v378_acc += ((v418_data[1]) * v122_data);
              v378_acc += ((v418_data[2]) * v127_data);
              v378_acc += ((v418_data[3]) * v132_data);
              v378_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v427_acc{};
              tensorforge::intel_esimd::simd<float, 16> v431_data;
              v431_data.copy_from(s1 + (120_i32));
              v427_acc += ((v431_data[0]) * v37_data);
              v427_acc += ((v431_data[1]) * v42_data);
              v427_acc += ((v431_data[2]) * v47_data);
              v427_acc += ((v431_data[3]) * v52_data);
              v427_acc += ((v431_data[4]) * v57_data);
              v427_acc += ((v431_data[5]) * v62_data);
              v427_acc += ((v431_data[6]) * v67_data);
              v427_acc += ((v431_data[7]) * v72_data);
              v427_acc += ((v431_data[8]) * v77_data);
              v427_acc += ((v431_data[9]) * v82_data);
              v427_acc += ((v431_data[10]) * v87_data);
              v427_acc += ((v431_data[11]) * v92_data);
              v427_acc += ((v431_data[12]) * v97_data);
              v427_acc += ((v431_data[13]) * v102_data);
              v427_acc += ((v431_data[14]) * v107_data);
              v427_acc += ((v431_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v467_data;
              v467_data.copy_from(s1 + (136_i32));
              v427_acc += ((v467_data[0]) * v117_data);
              v427_acc += ((v467_data[1]) * v122_data);
              v427_acc += ((v467_data[2]) * v127_data);
              v427_acc += ((v467_data[3]) * v132_data);
              v427_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v476_acc{};
              tensorforge::intel_esimd::simd<float, 16> v480_data;
              v480_data.copy_from(s1 + (140_i32));
              v476_acc += ((v480_data[0]) * v37_data);
              v476_acc += ((v480_data[1]) * v42_data);
              v476_acc += ((v480_data[2]) * v47_data);
              v476_acc += ((v480_data[3]) * v52_data);
              v476_acc += ((v480_data[4]) * v57_data);
              v476_acc += ((v480_data[5]) * v62_data);
              v476_acc += ((v480_data[6]) * v67_data);
              v476_acc += ((v480_data[7]) * v72_data);
              v476_acc += ((v480_data[8]) * v77_data);
              v476_acc += ((v480_data[9]) * v82_data);
              v476_acc += ((v480_data[10]) * v87_data);
              v476_acc += ((v480_data[11]) * v92_data);
              v476_acc += ((v480_data[12]) * v97_data);
              v476_acc += ((v480_data[13]) * v102_data);
              v476_acc += ((v480_data[14]) * v107_data);
              v476_acc += ((v480_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v516_data;
              v516_data.copy_from(s1 + (156_i32));
              v476_acc += ((v516_data[0]) * v117_data);
              v476_acc += ((v516_data[1]) * v122_data);
              v476_acc += ((v516_data[2]) * v127_data);
              v476_acc += ((v516_data[3]) * v132_data);
              v476_acc.copy_to(ir0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v525_acc{};
              tensorforge::intel_esimd::simd<float, 16> v529_data;
              v529_data.copy_from(s1 + (160_i32));
              v525_acc += ((v529_data[0]) * v37_data);
              v525_acc += ((v529_data[1]) * v42_data);
              v525_acc += ((v529_data[2]) * v47_data);
              v525_acc += ((v529_data[3]) * v52_data);
              v525_acc += ((v529_data[4]) * v57_data);
              v525_acc += ((v529_data[5]) * v62_data);
              v525_acc += ((v529_data[6]) * v67_data);
              v525_acc += ((v529_data[7]) * v72_data);
              v525_acc += ((v529_data[8]) * v77_data);
              v525_acc += ((v529_data[9]) * v82_data);
              v525_acc += ((v529_data[10]) * v87_data);
              v525_acc += ((v529_data[11]) * v92_data);
              v525_acc += ((v529_data[12]) * v97_data);
              v525_acc += ((v529_data[13]) * v102_data);
              v525_acc += ((v529_data[14]) * v107_data);
              v525_acc += ((v529_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v565_data;
              v565_data.copy_from(s1 + (176_i32));
              v525_acc += ((v565_data[0]) * v117_data);
              v525_acc += ((v565_data[1]) * v122_data);
              v525_acc += ((v565_data[2]) * v127_data);
              v525_acc += ((v565_data[3]) * v132_data);
              v525_acc.copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v574_acc{};
              tensorforge::intel_esimd::simd<float, 16> v578_data;
              v578_data.copy_from(s1 + (180_i32));
              v574_acc += ((v578_data[0]) * v37_data);
              v574_acc += ((v578_data[1]) * v42_data);
              v574_acc += ((v578_data[2]) * v47_data);
              v574_acc += ((v578_data[3]) * v52_data);
              v574_acc += ((v578_data[4]) * v57_data);
              v574_acc += ((v578_data[5]) * v62_data);
              v574_acc += ((v578_data[6]) * v67_data);
              v574_acc += ((v578_data[7]) * v72_data);
              v574_acc += ((v578_data[8]) * v77_data);
              v574_acc += ((v578_data[9]) * v82_data);
              v574_acc += ((v578_data[10]) * v87_data);
              v574_acc += ((v578_data[11]) * v92_data);
              v574_acc += ((v578_data[12]) * v97_data);
              v574_acc += ((v578_data[13]) * v102_data);
              v574_acc += ((v578_data[14]) * v107_data);
              v574_acc += ((v578_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v614_data;
              v614_data.copy_from(s1 + (196_i32));
              v574_acc += ((v614_data[0]) * v117_data);
              v574_acc += ((v614_data[1]) * v122_data);
              v574_acc += ((v614_data[2]) * v127_data);
              v574_acc += ((v614_data[3]) * v132_data);
              v574_acc.copy_to(ir0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v623_acc{};
              tensorforge::intel_esimd::simd<float, 16> v627_data;
              v627_data.copy_from(s1 + (200_i32));
              v623_acc += ((v627_data[0]) * v37_data);
              v623_acc += ((v627_data[1]) * v42_data);
              v623_acc += ((v627_data[2]) * v47_data);
              v623_acc += ((v627_data[3]) * v52_data);
              v623_acc += ((v627_data[4]) * v57_data);
              v623_acc += ((v627_data[5]) * v62_data);
              v623_acc += ((v627_data[6]) * v67_data);
              v623_acc += ((v627_data[7]) * v72_data);
              v623_acc += ((v627_data[8]) * v77_data);
              v623_acc += ((v627_data[9]) * v82_data);
              v623_acc += ((v627_data[10]) * v87_data);
              v623_acc += ((v627_data[11]) * v92_data);
              v623_acc += ((v627_data[12]) * v97_data);
              v623_acc += ((v627_data[13]) * v102_data);
              v623_acc += ((v627_data[14]) * v107_data);
              v623_acc += ((v627_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v663_data;
              v663_data.copy_from(s1 + (216_i32));
              v623_acc += ((v663_data[0]) * v117_data);
              v623_acc += ((v663_data[1]) * v122_data);
              v623_acc += ((v663_data[2]) * v127_data);
              v623_acc += ((v663_data[3]) * v132_data);
              v623_acc.copy_to(ir0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v672_acc{};
              tensorforge::intel_esimd::simd<float, 16> v676_data;
              v676_data.copy_from(s1 + (220_i32));
              v672_acc += ((v676_data[0]) * v37_data);
              v672_acc += ((v676_data[1]) * v42_data);
              v672_acc += ((v676_data[2]) * v47_data);
              v672_acc += ((v676_data[3]) * v52_data);
              v672_acc += ((v676_data[4]) * v57_data);
              v672_acc += ((v676_data[5]) * v62_data);
              v672_acc += ((v676_data[6]) * v67_data);
              v672_acc += ((v676_data[7]) * v72_data);
              v672_acc += ((v676_data[8]) * v77_data);
              v672_acc += ((v676_data[9]) * v82_data);
              v672_acc += ((v676_data[10]) * v87_data);
              v672_acc += ((v676_data[11]) * v92_data);
              v672_acc += ((v676_data[12]) * v97_data);
              v672_acc += ((v676_data[13]) * v102_data);
              v672_acc += ((v676_data[14]) * v107_data);
              v672_acc += ((v676_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v712_data;
              v712_data.copy_from(s1 + (236_i32));
              v672_acc += ((v712_data[0]) * v117_data);
              v672_acc += ((v712_data[1]) * v122_data);
              v672_acc += ((v712_data[2]) * v127_data);
              v672_acc += ((v712_data[3]) * v132_data);
              v672_acc.copy_to(ir0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v721_acc{};
              tensorforge::intel_esimd::simd<float, 16> v725_data;
              v725_data.copy_from(s1 + (240_i32));
              v721_acc += ((v725_data[0]) * v37_data);
              v721_acc += ((v725_data[1]) * v42_data);
              v721_acc += ((v725_data[2]) * v47_data);
              v721_acc += ((v725_data[3]) * v52_data);
              v721_acc += ((v725_data[4]) * v57_data);
              v721_acc += ((v725_data[5]) * v62_data);
              v721_acc += ((v725_data[6]) * v67_data);
              v721_acc += ((v725_data[7]) * v72_data);
              v721_acc += ((v725_data[8]) * v77_data);
              v721_acc += ((v725_data[9]) * v82_data);
              v721_acc += ((v725_data[10]) * v87_data);
              v721_acc += ((v725_data[11]) * v92_data);
              v721_acc += ((v725_data[12]) * v97_data);
              v721_acc += ((v725_data[13]) * v102_data);
              v721_acc += ((v725_data[14]) * v107_data);
              v721_acc += ((v725_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v761_data;
              v761_data.copy_from(s1 + (256_i32));
              v721_acc += ((v761_data[0]) * v117_data);
              v721_acc += ((v761_data[1]) * v122_data);
              v721_acc += ((v761_data[2]) * v127_data);
              v721_acc += ((v761_data[3]) * v132_data);
              v721_acc.copy_to(ir0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v770_acc{};
              tensorforge::intel_esimd::simd<float, 16> v774_data;
              v774_data.copy_from(s1 + (260_i32));
              v770_acc += ((v774_data[0]) * v37_data);
              v770_acc += ((v774_data[1]) * v42_data);
              v770_acc += ((v774_data[2]) * v47_data);
              v770_acc += ((v774_data[3]) * v52_data);
              v770_acc += ((v774_data[4]) * v57_data);
              v770_acc += ((v774_data[5]) * v62_data);
              v770_acc += ((v774_data[6]) * v67_data);
              v770_acc += ((v774_data[7]) * v72_data);
              v770_acc += ((v774_data[8]) * v77_data);
              v770_acc += ((v774_data[9]) * v82_data);
              v770_acc += ((v774_data[10]) * v87_data);
              v770_acc += ((v774_data[11]) * v92_data);
              v770_acc += ((v774_data[12]) * v97_data);
              v770_acc += ((v774_data[13]) * v102_data);
              v770_acc += ((v774_data[14]) * v107_data);
              v770_acc += ((v774_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v810_data;
              v810_data.copy_from(s1 + (276_i32));
              v770_acc += ((v810_data[0]) * v117_data);
              v770_acc += ((v810_data[1]) * v122_data);
              v770_acc += ((v810_data[2]) * v127_data);
              v770_acc += ((v810_data[3]) * v132_data);
              v770_acc.copy_to(ir0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v819_acc{};
              tensorforge::intel_esimd::simd<float, 16> v823_data;
              v823_data.copy_from(s1 + (280_i32));
              v819_acc += ((v823_data[0]) * v37_data);
              v819_acc += ((v823_data[1]) * v42_data);
              v819_acc += ((v823_data[2]) * v47_data);
              v819_acc += ((v823_data[3]) * v52_data);
              v819_acc += ((v823_data[4]) * v57_data);
              v819_acc += ((v823_data[5]) * v62_data);
              v819_acc += ((v823_data[6]) * v67_data);
              v819_acc += ((v823_data[7]) * v72_data);
              v819_acc += ((v823_data[8]) * v77_data);
              v819_acc += ((v823_data[9]) * v82_data);
              v819_acc += ((v823_data[10]) * v87_data);
              v819_acc += ((v823_data[11]) * v92_data);
              v819_acc += ((v823_data[12]) * v97_data);
              v819_acc += ((v823_data[13]) * v102_data);
              v819_acc += ((v823_data[14]) * v107_data);
              v819_acc += ((v823_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v859_data;
              v859_data.copy_from(s1 + (296_i32));
              v819_acc += ((v859_data[0]) * v117_data);
              v819_acc += ((v859_data[1]) * v122_data);
              v819_acc += ((v859_data[2]) * v127_data);
              v819_acc += ((v859_data[3]) * v132_data);
              v819_acc.copy_to(ir0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v868_acc{};
              tensorforge::intel_esimd::simd<float, 16> v872_data;
              v872_data.copy_from(s1 + (300_i32));
              v868_acc += ((v872_data[0]) * v37_data);
              v868_acc += ((v872_data[1]) * v42_data);
              v868_acc += ((v872_data[2]) * v47_data);
              v868_acc += ((v872_data[3]) * v52_data);
              v868_acc += ((v872_data[4]) * v57_data);
              v868_acc += ((v872_data[5]) * v62_data);
              v868_acc += ((v872_data[6]) * v67_data);
              v868_acc += ((v872_data[7]) * v72_data);
              v868_acc += ((v872_data[8]) * v77_data);
              v868_acc += ((v872_data[9]) * v82_data);
              v868_acc += ((v872_data[10]) * v87_data);
              v868_acc += ((v872_data[11]) * v92_data);
              v868_acc += ((v872_data[12]) * v97_data);
              v868_acc += ((v872_data[13]) * v102_data);
              v868_acc += ((v872_data[14]) * v107_data);
              v868_acc += ((v872_data[15]) * v112_data);
              tensorforge::intel_esimd::simd<float, 16> v908_data;
              v908_data.copy_from(s1 + (316_i32));
              v868_acc += ((v908_data[0]) * v117_data);
              v868_acc += ((v908_data[1]) * v122_data);
              v868_acc += ((v908_data[2]) * v127_data);
              v868_acc += ((v908_data[3]) * v132_data);
              v868_acc.copy_to(ir0 + (240));
              #pragma unroll
              for (int32_t v917_n1 = 0; v917_n1 < 16; ++v917_n1) {
                int32_t v918_a = v917_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v920_data;
                v920_data.copy_from(ir0 + (v918_a));
                v920_data.copy_to(r0 + (v918_a));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v923_i1 = 0; v923_i1 < 16; ++v923_i1) {
                tensorforge::intel_esimd::simd<float, 12> v926_data;
                v926_data.copy_from(r0 + ((v923_i1 * 16)));
                v926_data.copy_to(glb_m0 + ((v923_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

