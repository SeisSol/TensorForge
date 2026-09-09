// === base name ===
kernel_9cfbc23a68a6d0eb

// === header ===
void launcher_kernel_9cfbc23a68a6d0eb(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_9cfbc23a68a6d0eb(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_9cfbc23a68a6d0eb(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_9cfbc23a68a6d0eb(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (9472, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          float* __restrict__ s0 = &localShrMem0[320];
          float* __restrict__ s1 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 320 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m1[1, 0])
              #pragma unroll
              for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
                int32_t v13_lead = v11_i0 * 16;
                #pragma unroll
                for (int32_t v12_i1 = 0; v12_i1 < 12; ++v12_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v17_data;
                  v17_data.copy_from(glb_m1 + ((v13_lead + (v12_i1 * 20))));
                  v17_data.copy_to(s0 + ((v13_lead + (v12_i1 * 21))));
                }
              }
              #pragma unroll
              for (int32_t v22_i1 = 0; v22_i1 < 12; ++v22_i1) {
                tensorforge::intel_esimd::simd<float, 4> v28_data;
                v28_data.copy_from(glb_m1 + ((16_i32 + (v22_i1 * 20))));
                v28_data.copy_to(s0 + ((16_i32 + (v22_i1 * 21))));
              }
              // s1 = load{g>s}(glb_m2[0, 1])
              #pragma unroll
              for (int32_t i = 0; i < 20; i += 4) {
                tensorforge::intel_esimd::simd<float, 64> v34_ld;
                v34_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + i * 16));
                v34_ld.copy_to(s1 + (0 + 0 + 4 * item.get_local_id(0) + i * 16));
              }
              // wait(s0 = load{g>s}(glb_m1[1, 0]));
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              float r0[256]{};
              // r0 = +(s0 * s1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir0[256]{};
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(s0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v46_data;
              v46_data.copy_from(s0 + (1_i32));
              tensorforge::intel_esimd::simd<float, 16> v51_data;
              v51_data.copy_from(s0 + (2_i32));
              tensorforge::intel_esimd::simd<float, 16> v56_data;
              v56_data.copy_from(s0 + (3_i32));
              tensorforge::intel_esimd::simd<float, 16> v61_data;
              v61_data.copy_from(s0 + (4_i32));
              tensorforge::intel_esimd::simd<float, 16> v66_data;
              v66_data.copy_from(s0 + (5_i32));
              tensorforge::intel_esimd::simd<float, 16> v71_data;
              v71_data.copy_from(s0 + (6_i32));
              tensorforge::intel_esimd::simd<float, 16> v76_data;
              v76_data.copy_from(s0 + (7_i32));
              tensorforge::intel_esimd::simd<float, 16> v81_data;
              v81_data.copy_from(s0 + (8_i32));
              tensorforge::intel_esimd::simd<float, 16> v86_data;
              v86_data.copy_from(s0 + (9_i32));
              tensorforge::intel_esimd::simd<float, 16> v91_data;
              v91_data.copy_from(s0 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v96_data;
              v96_data.copy_from(s0 + (11_i32));
              tensorforge::intel_esimd::simd<float, 16> v101_data;
              v101_data.copy_from(s0 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v106_data;
              v106_data.copy_from(s0 + (13_i32));
              tensorforge::intel_esimd::simd<float, 16> v111_data;
              v111_data.copy_from(s0 + (14_i32));
              tensorforge::intel_esimd::simd<float, 16> v116_data;
              v116_data.copy_from(s0 + (15_i32));
              tensorforge::intel_esimd::simd<float, 16> v121_data;
              v121_data.copy_from(s0 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v126_data;
              v126_data.copy_from(s0 + (17_i32));
              tensorforge::intel_esimd::simd<float, 16> v131_data;
              v131_data.copy_from(s0 + (18_i32));
              tensorforge::intel_esimd::simd<float, 16> v136_data;
              v136_data.copy_from(s0 + (19_i32));
              tensorforge::intel_esimd::simd<float, 16> v137_acc{};
              tensorforge::intel_esimd::simd<float, 16> v141_data;
              v141_data.copy_from(s1 + (0_i32));
              v137_acc += ((v141_data[0]) * v41_data);
              v137_acc += ((v141_data[1]) * v46_data);
              v137_acc += ((v141_data[2]) * v51_data);
              v137_acc += ((v141_data[3]) * v56_data);
              v137_acc += ((v141_data[4]) * v61_data);
              v137_acc += ((v141_data[5]) * v66_data);
              v137_acc += ((v141_data[6]) * v71_data);
              v137_acc += ((v141_data[7]) * v76_data);
              v137_acc += ((v141_data[8]) * v81_data);
              v137_acc += ((v141_data[9]) * v86_data);
              v137_acc += ((v141_data[10]) * v91_data);
              v137_acc += ((v141_data[11]) * v96_data);
              v137_acc += ((v141_data[12]) * v101_data);
              v137_acc += ((v141_data[13]) * v106_data);
              v137_acc += ((v141_data[14]) * v111_data);
              v137_acc += ((v141_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v177_data;
              v177_data.copy_from(s1 + (16_i32));
              v137_acc += ((v177_data[0]) * v121_data);
              v137_acc += ((v177_data[1]) * v126_data);
              v137_acc += ((v177_data[2]) * v131_data);
              v137_acc += ((v177_data[3]) * v136_data);
              v137_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v186_acc{};
              tensorforge::intel_esimd::simd<float, 16> v190_data;
              v190_data.copy_from(s1 + (20_i32));
              v186_acc += ((v190_data[0]) * v41_data);
              v186_acc += ((v190_data[1]) * v46_data);
              v186_acc += ((v190_data[2]) * v51_data);
              v186_acc += ((v190_data[3]) * v56_data);
              v186_acc += ((v190_data[4]) * v61_data);
              v186_acc += ((v190_data[5]) * v66_data);
              v186_acc += ((v190_data[6]) * v71_data);
              v186_acc += ((v190_data[7]) * v76_data);
              v186_acc += ((v190_data[8]) * v81_data);
              v186_acc += ((v190_data[9]) * v86_data);
              v186_acc += ((v190_data[10]) * v91_data);
              v186_acc += ((v190_data[11]) * v96_data);
              v186_acc += ((v190_data[12]) * v101_data);
              v186_acc += ((v190_data[13]) * v106_data);
              v186_acc += ((v190_data[14]) * v111_data);
              v186_acc += ((v190_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v226_data;
              v226_data.copy_from(s1 + (36_i32));
              v186_acc += ((v226_data[0]) * v121_data);
              v186_acc += ((v226_data[1]) * v126_data);
              v186_acc += ((v226_data[2]) * v131_data);
              v186_acc += ((v226_data[3]) * v136_data);
              v186_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v235_acc{};
              tensorforge::intel_esimd::simd<float, 16> v239_data;
              v239_data.copy_from(s1 + (40_i32));
              v235_acc += ((v239_data[0]) * v41_data);
              v235_acc += ((v239_data[1]) * v46_data);
              v235_acc += ((v239_data[2]) * v51_data);
              v235_acc += ((v239_data[3]) * v56_data);
              v235_acc += ((v239_data[4]) * v61_data);
              v235_acc += ((v239_data[5]) * v66_data);
              v235_acc += ((v239_data[6]) * v71_data);
              v235_acc += ((v239_data[7]) * v76_data);
              v235_acc += ((v239_data[8]) * v81_data);
              v235_acc += ((v239_data[9]) * v86_data);
              v235_acc += ((v239_data[10]) * v91_data);
              v235_acc += ((v239_data[11]) * v96_data);
              v235_acc += ((v239_data[12]) * v101_data);
              v235_acc += ((v239_data[13]) * v106_data);
              v235_acc += ((v239_data[14]) * v111_data);
              v235_acc += ((v239_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v275_data;
              v275_data.copy_from(s1 + (56_i32));
              v235_acc += ((v275_data[0]) * v121_data);
              v235_acc += ((v275_data[1]) * v126_data);
              v235_acc += ((v275_data[2]) * v131_data);
              v235_acc += ((v275_data[3]) * v136_data);
              v235_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v284_acc{};
              tensorforge::intel_esimd::simd<float, 16> v288_data;
              v288_data.copy_from(s1 + (60_i32));
              v284_acc += ((v288_data[0]) * v41_data);
              v284_acc += ((v288_data[1]) * v46_data);
              v284_acc += ((v288_data[2]) * v51_data);
              v284_acc += ((v288_data[3]) * v56_data);
              v284_acc += ((v288_data[4]) * v61_data);
              v284_acc += ((v288_data[5]) * v66_data);
              v284_acc += ((v288_data[6]) * v71_data);
              v284_acc += ((v288_data[7]) * v76_data);
              v284_acc += ((v288_data[8]) * v81_data);
              v284_acc += ((v288_data[9]) * v86_data);
              v284_acc += ((v288_data[10]) * v91_data);
              v284_acc += ((v288_data[11]) * v96_data);
              v284_acc += ((v288_data[12]) * v101_data);
              v284_acc += ((v288_data[13]) * v106_data);
              v284_acc += ((v288_data[14]) * v111_data);
              v284_acc += ((v288_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v324_data;
              v324_data.copy_from(s1 + (76_i32));
              v284_acc += ((v324_data[0]) * v121_data);
              v284_acc += ((v324_data[1]) * v126_data);
              v284_acc += ((v324_data[2]) * v131_data);
              v284_acc += ((v324_data[3]) * v136_data);
              v284_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v333_acc{};
              tensorforge::intel_esimd::simd<float, 16> v337_data;
              v337_data.copy_from(s1 + (80_i32));
              v333_acc += ((v337_data[0]) * v41_data);
              v333_acc += ((v337_data[1]) * v46_data);
              v333_acc += ((v337_data[2]) * v51_data);
              v333_acc += ((v337_data[3]) * v56_data);
              v333_acc += ((v337_data[4]) * v61_data);
              v333_acc += ((v337_data[5]) * v66_data);
              v333_acc += ((v337_data[6]) * v71_data);
              v333_acc += ((v337_data[7]) * v76_data);
              v333_acc += ((v337_data[8]) * v81_data);
              v333_acc += ((v337_data[9]) * v86_data);
              v333_acc += ((v337_data[10]) * v91_data);
              v333_acc += ((v337_data[11]) * v96_data);
              v333_acc += ((v337_data[12]) * v101_data);
              v333_acc += ((v337_data[13]) * v106_data);
              v333_acc += ((v337_data[14]) * v111_data);
              v333_acc += ((v337_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v373_data;
              v373_data.copy_from(s1 + (96_i32));
              v333_acc += ((v373_data[0]) * v121_data);
              v333_acc += ((v373_data[1]) * v126_data);
              v333_acc += ((v373_data[2]) * v131_data);
              v333_acc += ((v373_data[3]) * v136_data);
              v333_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v382_acc{};
              tensorforge::intel_esimd::simd<float, 16> v386_data;
              v386_data.copy_from(s1 + (100_i32));
              v382_acc += ((v386_data[0]) * v41_data);
              v382_acc += ((v386_data[1]) * v46_data);
              v382_acc += ((v386_data[2]) * v51_data);
              v382_acc += ((v386_data[3]) * v56_data);
              v382_acc += ((v386_data[4]) * v61_data);
              v382_acc += ((v386_data[5]) * v66_data);
              v382_acc += ((v386_data[6]) * v71_data);
              v382_acc += ((v386_data[7]) * v76_data);
              v382_acc += ((v386_data[8]) * v81_data);
              v382_acc += ((v386_data[9]) * v86_data);
              v382_acc += ((v386_data[10]) * v91_data);
              v382_acc += ((v386_data[11]) * v96_data);
              v382_acc += ((v386_data[12]) * v101_data);
              v382_acc += ((v386_data[13]) * v106_data);
              v382_acc += ((v386_data[14]) * v111_data);
              v382_acc += ((v386_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v422_data;
              v422_data.copy_from(s1 + (116_i32));
              v382_acc += ((v422_data[0]) * v121_data);
              v382_acc += ((v422_data[1]) * v126_data);
              v382_acc += ((v422_data[2]) * v131_data);
              v382_acc += ((v422_data[3]) * v136_data);
              v382_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v431_acc{};
              tensorforge::intel_esimd::simd<float, 16> v435_data;
              v435_data.copy_from(s1 + (120_i32));
              v431_acc += ((v435_data[0]) * v41_data);
              v431_acc += ((v435_data[1]) * v46_data);
              v431_acc += ((v435_data[2]) * v51_data);
              v431_acc += ((v435_data[3]) * v56_data);
              v431_acc += ((v435_data[4]) * v61_data);
              v431_acc += ((v435_data[5]) * v66_data);
              v431_acc += ((v435_data[6]) * v71_data);
              v431_acc += ((v435_data[7]) * v76_data);
              v431_acc += ((v435_data[8]) * v81_data);
              v431_acc += ((v435_data[9]) * v86_data);
              v431_acc += ((v435_data[10]) * v91_data);
              v431_acc += ((v435_data[11]) * v96_data);
              v431_acc += ((v435_data[12]) * v101_data);
              v431_acc += ((v435_data[13]) * v106_data);
              v431_acc += ((v435_data[14]) * v111_data);
              v431_acc += ((v435_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v471_data;
              v471_data.copy_from(s1 + (136_i32));
              v431_acc += ((v471_data[0]) * v121_data);
              v431_acc += ((v471_data[1]) * v126_data);
              v431_acc += ((v471_data[2]) * v131_data);
              v431_acc += ((v471_data[3]) * v136_data);
              v431_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v480_acc{};
              tensorforge::intel_esimd::simd<float, 16> v484_data;
              v484_data.copy_from(s1 + (140_i32));
              v480_acc += ((v484_data[0]) * v41_data);
              v480_acc += ((v484_data[1]) * v46_data);
              v480_acc += ((v484_data[2]) * v51_data);
              v480_acc += ((v484_data[3]) * v56_data);
              v480_acc += ((v484_data[4]) * v61_data);
              v480_acc += ((v484_data[5]) * v66_data);
              v480_acc += ((v484_data[6]) * v71_data);
              v480_acc += ((v484_data[7]) * v76_data);
              v480_acc += ((v484_data[8]) * v81_data);
              v480_acc += ((v484_data[9]) * v86_data);
              v480_acc += ((v484_data[10]) * v91_data);
              v480_acc += ((v484_data[11]) * v96_data);
              v480_acc += ((v484_data[12]) * v101_data);
              v480_acc += ((v484_data[13]) * v106_data);
              v480_acc += ((v484_data[14]) * v111_data);
              v480_acc += ((v484_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v520_data;
              v520_data.copy_from(s1 + (156_i32));
              v480_acc += ((v520_data[0]) * v121_data);
              v480_acc += ((v520_data[1]) * v126_data);
              v480_acc += ((v520_data[2]) * v131_data);
              v480_acc += ((v520_data[3]) * v136_data);
              v480_acc.copy_to(ir0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v529_acc{};
              tensorforge::intel_esimd::simd<float, 16> v533_data;
              v533_data.copy_from(s1 + (160_i32));
              v529_acc += ((v533_data[0]) * v41_data);
              v529_acc += ((v533_data[1]) * v46_data);
              v529_acc += ((v533_data[2]) * v51_data);
              v529_acc += ((v533_data[3]) * v56_data);
              v529_acc += ((v533_data[4]) * v61_data);
              v529_acc += ((v533_data[5]) * v66_data);
              v529_acc += ((v533_data[6]) * v71_data);
              v529_acc += ((v533_data[7]) * v76_data);
              v529_acc += ((v533_data[8]) * v81_data);
              v529_acc += ((v533_data[9]) * v86_data);
              v529_acc += ((v533_data[10]) * v91_data);
              v529_acc += ((v533_data[11]) * v96_data);
              v529_acc += ((v533_data[12]) * v101_data);
              v529_acc += ((v533_data[13]) * v106_data);
              v529_acc += ((v533_data[14]) * v111_data);
              v529_acc += ((v533_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v569_data;
              v569_data.copy_from(s1 + (176_i32));
              v529_acc += ((v569_data[0]) * v121_data);
              v529_acc += ((v569_data[1]) * v126_data);
              v529_acc += ((v569_data[2]) * v131_data);
              v529_acc += ((v569_data[3]) * v136_data);
              v529_acc.copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v578_acc{};
              tensorforge::intel_esimd::simd<float, 16> v582_data;
              v582_data.copy_from(s1 + (180_i32));
              v578_acc += ((v582_data[0]) * v41_data);
              v578_acc += ((v582_data[1]) * v46_data);
              v578_acc += ((v582_data[2]) * v51_data);
              v578_acc += ((v582_data[3]) * v56_data);
              v578_acc += ((v582_data[4]) * v61_data);
              v578_acc += ((v582_data[5]) * v66_data);
              v578_acc += ((v582_data[6]) * v71_data);
              v578_acc += ((v582_data[7]) * v76_data);
              v578_acc += ((v582_data[8]) * v81_data);
              v578_acc += ((v582_data[9]) * v86_data);
              v578_acc += ((v582_data[10]) * v91_data);
              v578_acc += ((v582_data[11]) * v96_data);
              v578_acc += ((v582_data[12]) * v101_data);
              v578_acc += ((v582_data[13]) * v106_data);
              v578_acc += ((v582_data[14]) * v111_data);
              v578_acc += ((v582_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v618_data;
              v618_data.copy_from(s1 + (196_i32));
              v578_acc += ((v618_data[0]) * v121_data);
              v578_acc += ((v618_data[1]) * v126_data);
              v578_acc += ((v618_data[2]) * v131_data);
              v578_acc += ((v618_data[3]) * v136_data);
              v578_acc.copy_to(ir0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v627_acc{};
              tensorforge::intel_esimd::simd<float, 16> v631_data;
              v631_data.copy_from(s1 + (200_i32));
              v627_acc += ((v631_data[0]) * v41_data);
              v627_acc += ((v631_data[1]) * v46_data);
              v627_acc += ((v631_data[2]) * v51_data);
              v627_acc += ((v631_data[3]) * v56_data);
              v627_acc += ((v631_data[4]) * v61_data);
              v627_acc += ((v631_data[5]) * v66_data);
              v627_acc += ((v631_data[6]) * v71_data);
              v627_acc += ((v631_data[7]) * v76_data);
              v627_acc += ((v631_data[8]) * v81_data);
              v627_acc += ((v631_data[9]) * v86_data);
              v627_acc += ((v631_data[10]) * v91_data);
              v627_acc += ((v631_data[11]) * v96_data);
              v627_acc += ((v631_data[12]) * v101_data);
              v627_acc += ((v631_data[13]) * v106_data);
              v627_acc += ((v631_data[14]) * v111_data);
              v627_acc += ((v631_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v667_data;
              v667_data.copy_from(s1 + (216_i32));
              v627_acc += ((v667_data[0]) * v121_data);
              v627_acc += ((v667_data[1]) * v126_data);
              v627_acc += ((v667_data[2]) * v131_data);
              v627_acc += ((v667_data[3]) * v136_data);
              v627_acc.copy_to(ir0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v676_acc{};
              tensorforge::intel_esimd::simd<float, 16> v680_data;
              v680_data.copy_from(s1 + (220_i32));
              v676_acc += ((v680_data[0]) * v41_data);
              v676_acc += ((v680_data[1]) * v46_data);
              v676_acc += ((v680_data[2]) * v51_data);
              v676_acc += ((v680_data[3]) * v56_data);
              v676_acc += ((v680_data[4]) * v61_data);
              v676_acc += ((v680_data[5]) * v66_data);
              v676_acc += ((v680_data[6]) * v71_data);
              v676_acc += ((v680_data[7]) * v76_data);
              v676_acc += ((v680_data[8]) * v81_data);
              v676_acc += ((v680_data[9]) * v86_data);
              v676_acc += ((v680_data[10]) * v91_data);
              v676_acc += ((v680_data[11]) * v96_data);
              v676_acc += ((v680_data[12]) * v101_data);
              v676_acc += ((v680_data[13]) * v106_data);
              v676_acc += ((v680_data[14]) * v111_data);
              v676_acc += ((v680_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v716_data;
              v716_data.copy_from(s1 + (236_i32));
              v676_acc += ((v716_data[0]) * v121_data);
              v676_acc += ((v716_data[1]) * v126_data);
              v676_acc += ((v716_data[2]) * v131_data);
              v676_acc += ((v716_data[3]) * v136_data);
              v676_acc.copy_to(ir0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v725_acc{};
              tensorforge::intel_esimd::simd<float, 16> v729_data;
              v729_data.copy_from(s1 + (240_i32));
              v725_acc += ((v729_data[0]) * v41_data);
              v725_acc += ((v729_data[1]) * v46_data);
              v725_acc += ((v729_data[2]) * v51_data);
              v725_acc += ((v729_data[3]) * v56_data);
              v725_acc += ((v729_data[4]) * v61_data);
              v725_acc += ((v729_data[5]) * v66_data);
              v725_acc += ((v729_data[6]) * v71_data);
              v725_acc += ((v729_data[7]) * v76_data);
              v725_acc += ((v729_data[8]) * v81_data);
              v725_acc += ((v729_data[9]) * v86_data);
              v725_acc += ((v729_data[10]) * v91_data);
              v725_acc += ((v729_data[11]) * v96_data);
              v725_acc += ((v729_data[12]) * v101_data);
              v725_acc += ((v729_data[13]) * v106_data);
              v725_acc += ((v729_data[14]) * v111_data);
              v725_acc += ((v729_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v765_data;
              v765_data.copy_from(s1 + (256_i32));
              v725_acc += ((v765_data[0]) * v121_data);
              v725_acc += ((v765_data[1]) * v126_data);
              v725_acc += ((v765_data[2]) * v131_data);
              v725_acc += ((v765_data[3]) * v136_data);
              v725_acc.copy_to(ir0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v774_acc{};
              tensorforge::intel_esimd::simd<float, 16> v778_data;
              v778_data.copy_from(s1 + (260_i32));
              v774_acc += ((v778_data[0]) * v41_data);
              v774_acc += ((v778_data[1]) * v46_data);
              v774_acc += ((v778_data[2]) * v51_data);
              v774_acc += ((v778_data[3]) * v56_data);
              v774_acc += ((v778_data[4]) * v61_data);
              v774_acc += ((v778_data[5]) * v66_data);
              v774_acc += ((v778_data[6]) * v71_data);
              v774_acc += ((v778_data[7]) * v76_data);
              v774_acc += ((v778_data[8]) * v81_data);
              v774_acc += ((v778_data[9]) * v86_data);
              v774_acc += ((v778_data[10]) * v91_data);
              v774_acc += ((v778_data[11]) * v96_data);
              v774_acc += ((v778_data[12]) * v101_data);
              v774_acc += ((v778_data[13]) * v106_data);
              v774_acc += ((v778_data[14]) * v111_data);
              v774_acc += ((v778_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v814_data;
              v814_data.copy_from(s1 + (276_i32));
              v774_acc += ((v814_data[0]) * v121_data);
              v774_acc += ((v814_data[1]) * v126_data);
              v774_acc += ((v814_data[2]) * v131_data);
              v774_acc += ((v814_data[3]) * v136_data);
              v774_acc.copy_to(ir0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v823_acc{};
              tensorforge::intel_esimd::simd<float, 16> v827_data;
              v827_data.copy_from(s1 + (280_i32));
              v823_acc += ((v827_data[0]) * v41_data);
              v823_acc += ((v827_data[1]) * v46_data);
              v823_acc += ((v827_data[2]) * v51_data);
              v823_acc += ((v827_data[3]) * v56_data);
              v823_acc += ((v827_data[4]) * v61_data);
              v823_acc += ((v827_data[5]) * v66_data);
              v823_acc += ((v827_data[6]) * v71_data);
              v823_acc += ((v827_data[7]) * v76_data);
              v823_acc += ((v827_data[8]) * v81_data);
              v823_acc += ((v827_data[9]) * v86_data);
              v823_acc += ((v827_data[10]) * v91_data);
              v823_acc += ((v827_data[11]) * v96_data);
              v823_acc += ((v827_data[12]) * v101_data);
              v823_acc += ((v827_data[13]) * v106_data);
              v823_acc += ((v827_data[14]) * v111_data);
              v823_acc += ((v827_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v863_data;
              v863_data.copy_from(s1 + (296_i32));
              v823_acc += ((v863_data[0]) * v121_data);
              v823_acc += ((v863_data[1]) * v126_data);
              v823_acc += ((v863_data[2]) * v131_data);
              v823_acc += ((v863_data[3]) * v136_data);
              v823_acc.copy_to(ir0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v872_acc{};
              tensorforge::intel_esimd::simd<float, 16> v876_data;
              v876_data.copy_from(s1 + (300_i32));
              v872_acc += ((v876_data[0]) * v41_data);
              v872_acc += ((v876_data[1]) * v46_data);
              v872_acc += ((v876_data[2]) * v51_data);
              v872_acc += ((v876_data[3]) * v56_data);
              v872_acc += ((v876_data[4]) * v61_data);
              v872_acc += ((v876_data[5]) * v66_data);
              v872_acc += ((v876_data[6]) * v71_data);
              v872_acc += ((v876_data[7]) * v76_data);
              v872_acc += ((v876_data[8]) * v81_data);
              v872_acc += ((v876_data[9]) * v86_data);
              v872_acc += ((v876_data[10]) * v91_data);
              v872_acc += ((v876_data[11]) * v96_data);
              v872_acc += ((v876_data[12]) * v101_data);
              v872_acc += ((v876_data[13]) * v106_data);
              v872_acc += ((v876_data[14]) * v111_data);
              v872_acc += ((v876_data[15]) * v116_data);
              tensorforge::intel_esimd::simd<float, 16> v912_data;
              v912_data.copy_from(s1 + (316_i32));
              v872_acc += ((v912_data[0]) * v121_data);
              v872_acc += ((v912_data[1]) * v126_data);
              v872_acc += ((v912_data[2]) * v131_data);
              v872_acc += ((v912_data[3]) * v136_data);
              v872_acc.copy_to(ir0 + (240));
              #pragma unroll
              for (int32_t v921_n1 = 0; v921_n1 < 16; ++v921_n1) {
                int32_t v922_a = v921_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v924_data;
                v924_data.copy_from(ir0 + (v922_a));
                v924_data.copy_to(r0 + (v922_a));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v927_i1 = 0; v927_i1 < 16; ++v927_i1) {
                tensorforge::intel_esimd::simd<float, 12> v930_data;
                v930_data.copy_from(r0 + ((v927_i1 * 16)));
                v930_data.copy_to(glb_m0 + ((v927_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

