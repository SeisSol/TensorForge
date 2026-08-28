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
              for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
                int32_t v6_lead = v4_i0 * 16;
                #pragma unroll
                for (int32_t v5_i1 = 0; v5_i1 < 12; ++v5_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v10_data;
                  v10_data.copy_from(glb_m1 + ((v6_lead + (v5_i1 * 20))));
                  v10_data.copy_to(s0 + ((v6_lead + (v5_i1 * 21))));
                }
              }
              #pragma unroll
              for (int32_t v15_i1 = 0; v15_i1 < 12; ++v15_i1) {
                tensorforge::intel_esimd::simd<float, 4> v21_data;
                v21_data.copy_from(glb_m1 + ((16_i32 + (v15_i1 * 20))));
                v21_data.copy_to(s0 + ((16_i32 + (v15_i1 * 21))));
              }
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m2[0, 1])
              #pragma unroll
              for (int32_t i = 0; i < 20; i += 4) {
                *(sycl::vec<float, 4>*)&s1[0 + 0 + 4 * item.get_local_id(0) + i * 16] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + i * 16];
              }
              // wait(s0 = load{g>s}(glb_m1[1, 0]));
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              float r0[256]{};
              // r0 = +(s0 * s1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir0[256]{};
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(s0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(s0 + (1_i32));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(s0 + (2_i32));
              tensorforge::intel_esimd::simd<float, 16> v49_data;
              v49_data.copy_from(s0 + (3_i32));
              tensorforge::intel_esimd::simd<float, 16> v54_data;
              v54_data.copy_from(s0 + (4_i32));
              tensorforge::intel_esimd::simd<float, 16> v59_data;
              v59_data.copy_from(s0 + (5_i32));
              tensorforge::intel_esimd::simd<float, 16> v64_data;
              v64_data.copy_from(s0 + (6_i32));
              tensorforge::intel_esimd::simd<float, 16> v69_data;
              v69_data.copy_from(s0 + (7_i32));
              tensorforge::intel_esimd::simd<float, 16> v74_data;
              v74_data.copy_from(s0 + (8_i32));
              tensorforge::intel_esimd::simd<float, 16> v79_data;
              v79_data.copy_from(s0 + (9_i32));
              tensorforge::intel_esimd::simd<float, 16> v84_data;
              v84_data.copy_from(s0 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v89_data;
              v89_data.copy_from(s0 + (11_i32));
              tensorforge::intel_esimd::simd<float, 16> v94_data;
              v94_data.copy_from(s0 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v99_data;
              v99_data.copy_from(s0 + (13_i32));
              tensorforge::intel_esimd::simd<float, 16> v104_data;
              v104_data.copy_from(s0 + (14_i32));
              tensorforge::intel_esimd::simd<float, 16> v109_data;
              v109_data.copy_from(s0 + (15_i32));
              tensorforge::intel_esimd::simd<float, 16> v114_data;
              v114_data.copy_from(s0 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v119_data;
              v119_data.copy_from(s0 + (17_i32));
              tensorforge::intel_esimd::simd<float, 16> v124_data;
              v124_data.copy_from(s0 + (18_i32));
              tensorforge::intel_esimd::simd<float, 16> v129_data;
              v129_data.copy_from(s0 + (19_i32));
              tensorforge::intel_esimd::simd<float, 16> v130_acc{};
              tensorforge::intel_esimd::simd<float, 16> v134_data;
              v134_data.copy_from(s1 + (0_i32));
              v130_acc += ((v134_data[0]) * v34_data);
              v130_acc += ((v134_data[1]) * v39_data);
              v130_acc += ((v134_data[2]) * v44_data);
              v130_acc += ((v134_data[3]) * v49_data);
              v130_acc += ((v134_data[4]) * v54_data);
              v130_acc += ((v134_data[5]) * v59_data);
              v130_acc += ((v134_data[6]) * v64_data);
              v130_acc += ((v134_data[7]) * v69_data);
              v130_acc += ((v134_data[8]) * v74_data);
              v130_acc += ((v134_data[9]) * v79_data);
              v130_acc += ((v134_data[10]) * v84_data);
              v130_acc += ((v134_data[11]) * v89_data);
              v130_acc += ((v134_data[12]) * v94_data);
              v130_acc += ((v134_data[13]) * v99_data);
              v130_acc += ((v134_data[14]) * v104_data);
              v130_acc += ((v134_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v170_data;
              v170_data.copy_from(s1 + (16_i32));
              v130_acc += ((v170_data[0]) * v114_data);
              v130_acc += ((v170_data[1]) * v119_data);
              v130_acc += ((v170_data[2]) * v124_data);
              v130_acc += ((v170_data[3]) * v129_data);
              v130_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v179_acc{};
              tensorforge::intel_esimd::simd<float, 16> v183_data;
              v183_data.copy_from(s1 + (20_i32));
              v179_acc += ((v183_data[0]) * v34_data);
              v179_acc += ((v183_data[1]) * v39_data);
              v179_acc += ((v183_data[2]) * v44_data);
              v179_acc += ((v183_data[3]) * v49_data);
              v179_acc += ((v183_data[4]) * v54_data);
              v179_acc += ((v183_data[5]) * v59_data);
              v179_acc += ((v183_data[6]) * v64_data);
              v179_acc += ((v183_data[7]) * v69_data);
              v179_acc += ((v183_data[8]) * v74_data);
              v179_acc += ((v183_data[9]) * v79_data);
              v179_acc += ((v183_data[10]) * v84_data);
              v179_acc += ((v183_data[11]) * v89_data);
              v179_acc += ((v183_data[12]) * v94_data);
              v179_acc += ((v183_data[13]) * v99_data);
              v179_acc += ((v183_data[14]) * v104_data);
              v179_acc += ((v183_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v219_data;
              v219_data.copy_from(s1 + (36_i32));
              v179_acc += ((v219_data[0]) * v114_data);
              v179_acc += ((v219_data[1]) * v119_data);
              v179_acc += ((v219_data[2]) * v124_data);
              v179_acc += ((v219_data[3]) * v129_data);
              v179_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v228_acc{};
              tensorforge::intel_esimd::simd<float, 16> v232_data;
              v232_data.copy_from(s1 + (40_i32));
              v228_acc += ((v232_data[0]) * v34_data);
              v228_acc += ((v232_data[1]) * v39_data);
              v228_acc += ((v232_data[2]) * v44_data);
              v228_acc += ((v232_data[3]) * v49_data);
              v228_acc += ((v232_data[4]) * v54_data);
              v228_acc += ((v232_data[5]) * v59_data);
              v228_acc += ((v232_data[6]) * v64_data);
              v228_acc += ((v232_data[7]) * v69_data);
              v228_acc += ((v232_data[8]) * v74_data);
              v228_acc += ((v232_data[9]) * v79_data);
              v228_acc += ((v232_data[10]) * v84_data);
              v228_acc += ((v232_data[11]) * v89_data);
              v228_acc += ((v232_data[12]) * v94_data);
              v228_acc += ((v232_data[13]) * v99_data);
              v228_acc += ((v232_data[14]) * v104_data);
              v228_acc += ((v232_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v268_data;
              v268_data.copy_from(s1 + (56_i32));
              v228_acc += ((v268_data[0]) * v114_data);
              v228_acc += ((v268_data[1]) * v119_data);
              v228_acc += ((v268_data[2]) * v124_data);
              v228_acc += ((v268_data[3]) * v129_data);
              v228_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v277_acc{};
              tensorforge::intel_esimd::simd<float, 16> v281_data;
              v281_data.copy_from(s1 + (60_i32));
              v277_acc += ((v281_data[0]) * v34_data);
              v277_acc += ((v281_data[1]) * v39_data);
              v277_acc += ((v281_data[2]) * v44_data);
              v277_acc += ((v281_data[3]) * v49_data);
              v277_acc += ((v281_data[4]) * v54_data);
              v277_acc += ((v281_data[5]) * v59_data);
              v277_acc += ((v281_data[6]) * v64_data);
              v277_acc += ((v281_data[7]) * v69_data);
              v277_acc += ((v281_data[8]) * v74_data);
              v277_acc += ((v281_data[9]) * v79_data);
              v277_acc += ((v281_data[10]) * v84_data);
              v277_acc += ((v281_data[11]) * v89_data);
              v277_acc += ((v281_data[12]) * v94_data);
              v277_acc += ((v281_data[13]) * v99_data);
              v277_acc += ((v281_data[14]) * v104_data);
              v277_acc += ((v281_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v317_data;
              v317_data.copy_from(s1 + (76_i32));
              v277_acc += ((v317_data[0]) * v114_data);
              v277_acc += ((v317_data[1]) * v119_data);
              v277_acc += ((v317_data[2]) * v124_data);
              v277_acc += ((v317_data[3]) * v129_data);
              v277_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v326_acc{};
              tensorforge::intel_esimd::simd<float, 16> v330_data;
              v330_data.copy_from(s1 + (80_i32));
              v326_acc += ((v330_data[0]) * v34_data);
              v326_acc += ((v330_data[1]) * v39_data);
              v326_acc += ((v330_data[2]) * v44_data);
              v326_acc += ((v330_data[3]) * v49_data);
              v326_acc += ((v330_data[4]) * v54_data);
              v326_acc += ((v330_data[5]) * v59_data);
              v326_acc += ((v330_data[6]) * v64_data);
              v326_acc += ((v330_data[7]) * v69_data);
              v326_acc += ((v330_data[8]) * v74_data);
              v326_acc += ((v330_data[9]) * v79_data);
              v326_acc += ((v330_data[10]) * v84_data);
              v326_acc += ((v330_data[11]) * v89_data);
              v326_acc += ((v330_data[12]) * v94_data);
              v326_acc += ((v330_data[13]) * v99_data);
              v326_acc += ((v330_data[14]) * v104_data);
              v326_acc += ((v330_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v366_data;
              v366_data.copy_from(s1 + (96_i32));
              v326_acc += ((v366_data[0]) * v114_data);
              v326_acc += ((v366_data[1]) * v119_data);
              v326_acc += ((v366_data[2]) * v124_data);
              v326_acc += ((v366_data[3]) * v129_data);
              v326_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v375_acc{};
              tensorforge::intel_esimd::simd<float, 16> v379_data;
              v379_data.copy_from(s1 + (100_i32));
              v375_acc += ((v379_data[0]) * v34_data);
              v375_acc += ((v379_data[1]) * v39_data);
              v375_acc += ((v379_data[2]) * v44_data);
              v375_acc += ((v379_data[3]) * v49_data);
              v375_acc += ((v379_data[4]) * v54_data);
              v375_acc += ((v379_data[5]) * v59_data);
              v375_acc += ((v379_data[6]) * v64_data);
              v375_acc += ((v379_data[7]) * v69_data);
              v375_acc += ((v379_data[8]) * v74_data);
              v375_acc += ((v379_data[9]) * v79_data);
              v375_acc += ((v379_data[10]) * v84_data);
              v375_acc += ((v379_data[11]) * v89_data);
              v375_acc += ((v379_data[12]) * v94_data);
              v375_acc += ((v379_data[13]) * v99_data);
              v375_acc += ((v379_data[14]) * v104_data);
              v375_acc += ((v379_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v415_data;
              v415_data.copy_from(s1 + (116_i32));
              v375_acc += ((v415_data[0]) * v114_data);
              v375_acc += ((v415_data[1]) * v119_data);
              v375_acc += ((v415_data[2]) * v124_data);
              v375_acc += ((v415_data[3]) * v129_data);
              v375_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v424_acc{};
              tensorforge::intel_esimd::simd<float, 16> v428_data;
              v428_data.copy_from(s1 + (120_i32));
              v424_acc += ((v428_data[0]) * v34_data);
              v424_acc += ((v428_data[1]) * v39_data);
              v424_acc += ((v428_data[2]) * v44_data);
              v424_acc += ((v428_data[3]) * v49_data);
              v424_acc += ((v428_data[4]) * v54_data);
              v424_acc += ((v428_data[5]) * v59_data);
              v424_acc += ((v428_data[6]) * v64_data);
              v424_acc += ((v428_data[7]) * v69_data);
              v424_acc += ((v428_data[8]) * v74_data);
              v424_acc += ((v428_data[9]) * v79_data);
              v424_acc += ((v428_data[10]) * v84_data);
              v424_acc += ((v428_data[11]) * v89_data);
              v424_acc += ((v428_data[12]) * v94_data);
              v424_acc += ((v428_data[13]) * v99_data);
              v424_acc += ((v428_data[14]) * v104_data);
              v424_acc += ((v428_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v464_data;
              v464_data.copy_from(s1 + (136_i32));
              v424_acc += ((v464_data[0]) * v114_data);
              v424_acc += ((v464_data[1]) * v119_data);
              v424_acc += ((v464_data[2]) * v124_data);
              v424_acc += ((v464_data[3]) * v129_data);
              v424_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v473_acc{};
              tensorforge::intel_esimd::simd<float, 16> v477_data;
              v477_data.copy_from(s1 + (140_i32));
              v473_acc += ((v477_data[0]) * v34_data);
              v473_acc += ((v477_data[1]) * v39_data);
              v473_acc += ((v477_data[2]) * v44_data);
              v473_acc += ((v477_data[3]) * v49_data);
              v473_acc += ((v477_data[4]) * v54_data);
              v473_acc += ((v477_data[5]) * v59_data);
              v473_acc += ((v477_data[6]) * v64_data);
              v473_acc += ((v477_data[7]) * v69_data);
              v473_acc += ((v477_data[8]) * v74_data);
              v473_acc += ((v477_data[9]) * v79_data);
              v473_acc += ((v477_data[10]) * v84_data);
              v473_acc += ((v477_data[11]) * v89_data);
              v473_acc += ((v477_data[12]) * v94_data);
              v473_acc += ((v477_data[13]) * v99_data);
              v473_acc += ((v477_data[14]) * v104_data);
              v473_acc += ((v477_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v513_data;
              v513_data.copy_from(s1 + (156_i32));
              v473_acc += ((v513_data[0]) * v114_data);
              v473_acc += ((v513_data[1]) * v119_data);
              v473_acc += ((v513_data[2]) * v124_data);
              v473_acc += ((v513_data[3]) * v129_data);
              v473_acc.copy_to(ir0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v522_acc{};
              tensorforge::intel_esimd::simd<float, 16> v526_data;
              v526_data.copy_from(s1 + (160_i32));
              v522_acc += ((v526_data[0]) * v34_data);
              v522_acc += ((v526_data[1]) * v39_data);
              v522_acc += ((v526_data[2]) * v44_data);
              v522_acc += ((v526_data[3]) * v49_data);
              v522_acc += ((v526_data[4]) * v54_data);
              v522_acc += ((v526_data[5]) * v59_data);
              v522_acc += ((v526_data[6]) * v64_data);
              v522_acc += ((v526_data[7]) * v69_data);
              v522_acc += ((v526_data[8]) * v74_data);
              v522_acc += ((v526_data[9]) * v79_data);
              v522_acc += ((v526_data[10]) * v84_data);
              v522_acc += ((v526_data[11]) * v89_data);
              v522_acc += ((v526_data[12]) * v94_data);
              v522_acc += ((v526_data[13]) * v99_data);
              v522_acc += ((v526_data[14]) * v104_data);
              v522_acc += ((v526_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v562_data;
              v562_data.copy_from(s1 + (176_i32));
              v522_acc += ((v562_data[0]) * v114_data);
              v522_acc += ((v562_data[1]) * v119_data);
              v522_acc += ((v562_data[2]) * v124_data);
              v522_acc += ((v562_data[3]) * v129_data);
              v522_acc.copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v571_acc{};
              tensorforge::intel_esimd::simd<float, 16> v575_data;
              v575_data.copy_from(s1 + (180_i32));
              v571_acc += ((v575_data[0]) * v34_data);
              v571_acc += ((v575_data[1]) * v39_data);
              v571_acc += ((v575_data[2]) * v44_data);
              v571_acc += ((v575_data[3]) * v49_data);
              v571_acc += ((v575_data[4]) * v54_data);
              v571_acc += ((v575_data[5]) * v59_data);
              v571_acc += ((v575_data[6]) * v64_data);
              v571_acc += ((v575_data[7]) * v69_data);
              v571_acc += ((v575_data[8]) * v74_data);
              v571_acc += ((v575_data[9]) * v79_data);
              v571_acc += ((v575_data[10]) * v84_data);
              v571_acc += ((v575_data[11]) * v89_data);
              v571_acc += ((v575_data[12]) * v94_data);
              v571_acc += ((v575_data[13]) * v99_data);
              v571_acc += ((v575_data[14]) * v104_data);
              v571_acc += ((v575_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v611_data;
              v611_data.copy_from(s1 + (196_i32));
              v571_acc += ((v611_data[0]) * v114_data);
              v571_acc += ((v611_data[1]) * v119_data);
              v571_acc += ((v611_data[2]) * v124_data);
              v571_acc += ((v611_data[3]) * v129_data);
              v571_acc.copy_to(ir0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v620_acc{};
              tensorforge::intel_esimd::simd<float, 16> v624_data;
              v624_data.copy_from(s1 + (200_i32));
              v620_acc += ((v624_data[0]) * v34_data);
              v620_acc += ((v624_data[1]) * v39_data);
              v620_acc += ((v624_data[2]) * v44_data);
              v620_acc += ((v624_data[3]) * v49_data);
              v620_acc += ((v624_data[4]) * v54_data);
              v620_acc += ((v624_data[5]) * v59_data);
              v620_acc += ((v624_data[6]) * v64_data);
              v620_acc += ((v624_data[7]) * v69_data);
              v620_acc += ((v624_data[8]) * v74_data);
              v620_acc += ((v624_data[9]) * v79_data);
              v620_acc += ((v624_data[10]) * v84_data);
              v620_acc += ((v624_data[11]) * v89_data);
              v620_acc += ((v624_data[12]) * v94_data);
              v620_acc += ((v624_data[13]) * v99_data);
              v620_acc += ((v624_data[14]) * v104_data);
              v620_acc += ((v624_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v660_data;
              v660_data.copy_from(s1 + (216_i32));
              v620_acc += ((v660_data[0]) * v114_data);
              v620_acc += ((v660_data[1]) * v119_data);
              v620_acc += ((v660_data[2]) * v124_data);
              v620_acc += ((v660_data[3]) * v129_data);
              v620_acc.copy_to(ir0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v669_acc{};
              tensorforge::intel_esimd::simd<float, 16> v673_data;
              v673_data.copy_from(s1 + (220_i32));
              v669_acc += ((v673_data[0]) * v34_data);
              v669_acc += ((v673_data[1]) * v39_data);
              v669_acc += ((v673_data[2]) * v44_data);
              v669_acc += ((v673_data[3]) * v49_data);
              v669_acc += ((v673_data[4]) * v54_data);
              v669_acc += ((v673_data[5]) * v59_data);
              v669_acc += ((v673_data[6]) * v64_data);
              v669_acc += ((v673_data[7]) * v69_data);
              v669_acc += ((v673_data[8]) * v74_data);
              v669_acc += ((v673_data[9]) * v79_data);
              v669_acc += ((v673_data[10]) * v84_data);
              v669_acc += ((v673_data[11]) * v89_data);
              v669_acc += ((v673_data[12]) * v94_data);
              v669_acc += ((v673_data[13]) * v99_data);
              v669_acc += ((v673_data[14]) * v104_data);
              v669_acc += ((v673_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v709_data;
              v709_data.copy_from(s1 + (236_i32));
              v669_acc += ((v709_data[0]) * v114_data);
              v669_acc += ((v709_data[1]) * v119_data);
              v669_acc += ((v709_data[2]) * v124_data);
              v669_acc += ((v709_data[3]) * v129_data);
              v669_acc.copy_to(ir0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v718_acc{};
              tensorforge::intel_esimd::simd<float, 16> v722_data;
              v722_data.copy_from(s1 + (240_i32));
              v718_acc += ((v722_data[0]) * v34_data);
              v718_acc += ((v722_data[1]) * v39_data);
              v718_acc += ((v722_data[2]) * v44_data);
              v718_acc += ((v722_data[3]) * v49_data);
              v718_acc += ((v722_data[4]) * v54_data);
              v718_acc += ((v722_data[5]) * v59_data);
              v718_acc += ((v722_data[6]) * v64_data);
              v718_acc += ((v722_data[7]) * v69_data);
              v718_acc += ((v722_data[8]) * v74_data);
              v718_acc += ((v722_data[9]) * v79_data);
              v718_acc += ((v722_data[10]) * v84_data);
              v718_acc += ((v722_data[11]) * v89_data);
              v718_acc += ((v722_data[12]) * v94_data);
              v718_acc += ((v722_data[13]) * v99_data);
              v718_acc += ((v722_data[14]) * v104_data);
              v718_acc += ((v722_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v758_data;
              v758_data.copy_from(s1 + (256_i32));
              v718_acc += ((v758_data[0]) * v114_data);
              v718_acc += ((v758_data[1]) * v119_data);
              v718_acc += ((v758_data[2]) * v124_data);
              v718_acc += ((v758_data[3]) * v129_data);
              v718_acc.copy_to(ir0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v767_acc{};
              tensorforge::intel_esimd::simd<float, 16> v771_data;
              v771_data.copy_from(s1 + (260_i32));
              v767_acc += ((v771_data[0]) * v34_data);
              v767_acc += ((v771_data[1]) * v39_data);
              v767_acc += ((v771_data[2]) * v44_data);
              v767_acc += ((v771_data[3]) * v49_data);
              v767_acc += ((v771_data[4]) * v54_data);
              v767_acc += ((v771_data[5]) * v59_data);
              v767_acc += ((v771_data[6]) * v64_data);
              v767_acc += ((v771_data[7]) * v69_data);
              v767_acc += ((v771_data[8]) * v74_data);
              v767_acc += ((v771_data[9]) * v79_data);
              v767_acc += ((v771_data[10]) * v84_data);
              v767_acc += ((v771_data[11]) * v89_data);
              v767_acc += ((v771_data[12]) * v94_data);
              v767_acc += ((v771_data[13]) * v99_data);
              v767_acc += ((v771_data[14]) * v104_data);
              v767_acc += ((v771_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v807_data;
              v807_data.copy_from(s1 + (276_i32));
              v767_acc += ((v807_data[0]) * v114_data);
              v767_acc += ((v807_data[1]) * v119_data);
              v767_acc += ((v807_data[2]) * v124_data);
              v767_acc += ((v807_data[3]) * v129_data);
              v767_acc.copy_to(ir0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v816_acc{};
              tensorforge::intel_esimd::simd<float, 16> v820_data;
              v820_data.copy_from(s1 + (280_i32));
              v816_acc += ((v820_data[0]) * v34_data);
              v816_acc += ((v820_data[1]) * v39_data);
              v816_acc += ((v820_data[2]) * v44_data);
              v816_acc += ((v820_data[3]) * v49_data);
              v816_acc += ((v820_data[4]) * v54_data);
              v816_acc += ((v820_data[5]) * v59_data);
              v816_acc += ((v820_data[6]) * v64_data);
              v816_acc += ((v820_data[7]) * v69_data);
              v816_acc += ((v820_data[8]) * v74_data);
              v816_acc += ((v820_data[9]) * v79_data);
              v816_acc += ((v820_data[10]) * v84_data);
              v816_acc += ((v820_data[11]) * v89_data);
              v816_acc += ((v820_data[12]) * v94_data);
              v816_acc += ((v820_data[13]) * v99_data);
              v816_acc += ((v820_data[14]) * v104_data);
              v816_acc += ((v820_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v856_data;
              v856_data.copy_from(s1 + (296_i32));
              v816_acc += ((v856_data[0]) * v114_data);
              v816_acc += ((v856_data[1]) * v119_data);
              v816_acc += ((v856_data[2]) * v124_data);
              v816_acc += ((v856_data[3]) * v129_data);
              v816_acc.copy_to(ir0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v865_acc{};
              tensorforge::intel_esimd::simd<float, 16> v869_data;
              v869_data.copy_from(s1 + (300_i32));
              v865_acc += ((v869_data[0]) * v34_data);
              v865_acc += ((v869_data[1]) * v39_data);
              v865_acc += ((v869_data[2]) * v44_data);
              v865_acc += ((v869_data[3]) * v49_data);
              v865_acc += ((v869_data[4]) * v54_data);
              v865_acc += ((v869_data[5]) * v59_data);
              v865_acc += ((v869_data[6]) * v64_data);
              v865_acc += ((v869_data[7]) * v69_data);
              v865_acc += ((v869_data[8]) * v74_data);
              v865_acc += ((v869_data[9]) * v79_data);
              v865_acc += ((v869_data[10]) * v84_data);
              v865_acc += ((v869_data[11]) * v89_data);
              v865_acc += ((v869_data[12]) * v94_data);
              v865_acc += ((v869_data[13]) * v99_data);
              v865_acc += ((v869_data[14]) * v104_data);
              v865_acc += ((v869_data[15]) * v109_data);
              tensorforge::intel_esimd::simd<float, 16> v905_data;
              v905_data.copy_from(s1 + (316_i32));
              v865_acc += ((v905_data[0]) * v114_data);
              v865_acc += ((v905_data[1]) * v119_data);
              v865_acc += ((v905_data[2]) * v124_data);
              v865_acc += ((v905_data[3]) * v129_data);
              v865_acc.copy_to(ir0 + (240));
              #pragma unroll
              for (int32_t v914_n1 = 0; v914_n1 < 16; ++v914_n1) {
                int32_t v915_a = v914_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v917_data;
                v917_data.copy_from(ir0 + (v915_a));
                v917_data.copy_to(r0 + (v915_a));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v920_i1 = 0; v920_i1 < 16; ++v920_i1) {
                tensorforge::intel_esimd::simd<float, 12> v923_data;
                v923_data.copy_from(r0 + ((v920_i1 * 16)));
                v923_data.copy_to(glb_m0 + ((v920_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

