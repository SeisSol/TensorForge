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
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[8]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir0[8]{};
              tensorforge::intel_esimd::simd_mask<16> v7_g = (tensorforge::intel_esimd::simd<int32_t, 16>(0, 1)) < 12;
              int32_t v12_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v18_data(0.0f);
              v18_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[0_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v19_data(0.0f);
              v19_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[0]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v21_data(0.0f);
              v21_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v21_data + (v18_data * v19_data)).copy_to(ir0 + (0));
              }
              int32_t v27_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v34_data(0.0f);
              v34_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[16]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v36_data(0.0f);
              v36_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v36_data + (v18_data * v34_data)).copy_to(ir0 + (1));
              }
              int32_t v42_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v49_data(0.0f);
              v49_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v51_data(0.0f);
              v51_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v51_data + (v18_data * v49_data)).copy_to(ir0 + (2));
              }
              int32_t v57_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v64_data(0.0f);
              v64_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[48]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v66_data(0.0f);
              v66_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v66_data + (v18_data * v64_data)).copy_to(ir0 + (3));
              }
              int32_t v72_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v79_data(0.0f);
              v79_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[64]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v81_data(0.0f);
              v81_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v81_data + (v18_data * v79_data)).copy_to(ir0 + (4));
              }
              int32_t v87_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v94_data(0.0f);
              v94_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[80]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v96_data(0.0f);
              v96_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v96_data + (v18_data * v94_data)).copy_to(ir0 + (5));
              }
              int32_t v102_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v109_data(0.0f);
              v109_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[96]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v111_data(0.0f);
              v111_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v111_data + (v18_data * v109_data)).copy_to(ir0 + (6));
              }
              int32_t v117_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v124_data(0.0f);
              v124_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[112]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v126_data(0.0f);
              v126_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v126_data + (v18_data * v124_data)).copy_to(ir0 + (7));
              }
              int32_t v134_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v140_data(0.0f);
              v140_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[12_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v141_data(0.0f);
              v141_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[1]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v143_data(0.0f);
              v143_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v143_data + (v140_data * v141_data)).copy_to(ir0 + (0));
              }
              int32_t v149_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v156_data(0.0f);
              v156_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[17]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v158_data(0.0f);
              v158_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v158_data + (v140_data * v156_data)).copy_to(ir0 + (1));
              }
              int32_t v164_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v171_data(0.0f);
              v171_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[33]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v173_data(0.0f);
              v173_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v173_data + (v140_data * v171_data)).copy_to(ir0 + (2));
              }
              int32_t v179_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v186_data(0.0f);
              v186_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[49]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v188_data(0.0f);
              v188_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v188_data + (v140_data * v186_data)).copy_to(ir0 + (3));
              }
              int32_t v194_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v201_data(0.0f);
              v201_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[65]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v203_data(0.0f);
              v203_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v203_data + (v140_data * v201_data)).copy_to(ir0 + (4));
              }
              int32_t v209_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v216_data(0.0f);
              v216_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[81]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v218_data(0.0f);
              v218_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v218_data + (v140_data * v216_data)).copy_to(ir0 + (5));
              }
              int32_t v224_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v231_data(0.0f);
              v231_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[97]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v233_data(0.0f);
              v233_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v233_data + (v140_data * v231_data)).copy_to(ir0 + (6));
              }
              int32_t v239_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v246_data(0.0f);
              v246_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[113]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v248_data(0.0f);
              v248_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v248_data + (v140_data * v246_data)).copy_to(ir0 + (7));
              }
              int32_t v256_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v262_data(0.0f);
              v262_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[24_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v263_data(0.0f);
              v263_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[2]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v265_data(0.0f);
              v265_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v265_data + (v262_data * v263_data)).copy_to(ir0 + (0));
              }
              int32_t v271_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v278_data(0.0f);
              v278_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[18]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v280_data(0.0f);
              v280_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v280_data + (v262_data * v278_data)).copy_to(ir0 + (1));
              }
              int32_t v286_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v293_data(0.0f);
              v293_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[34]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v295_data(0.0f);
              v295_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v295_data + (v262_data * v293_data)).copy_to(ir0 + (2));
              }
              int32_t v301_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v308_data(0.0f);
              v308_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[50]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v310_data(0.0f);
              v310_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v310_data + (v262_data * v308_data)).copy_to(ir0 + (3));
              }
              int32_t v316_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v323_data(0.0f);
              v323_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[66]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v325_data(0.0f);
              v325_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v325_data + (v262_data * v323_data)).copy_to(ir0 + (4));
              }
              int32_t v331_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v338_data(0.0f);
              v338_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[82]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v340_data(0.0f);
              v340_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v340_data + (v262_data * v338_data)).copy_to(ir0 + (5));
              }
              int32_t v346_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v353_data(0.0f);
              v353_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[98]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v355_data(0.0f);
              v355_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v355_data + (v262_data * v353_data)).copy_to(ir0 + (6));
              }
              int32_t v361_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v368_data(0.0f);
              v368_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[114]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v370_data(0.0f);
              v370_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v370_data + (v262_data * v368_data)).copy_to(ir0 + (7));
              }
              int32_t v378_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v384_data(0.0f);
              v384_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[36_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v385_data(0.0f);
              v385_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[3]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v387_data(0.0f);
              v387_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v387_data + (v384_data * v385_data)).copy_to(ir0 + (0));
              }
              int32_t v393_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v400_data(0.0f);
              v400_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[19]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v402_data(0.0f);
              v402_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v402_data + (v384_data * v400_data)).copy_to(ir0 + (1));
              }
              int32_t v408_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v415_data(0.0f);
              v415_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[35]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v417_data(0.0f);
              v417_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v417_data + (v384_data * v415_data)).copy_to(ir0 + (2));
              }
              int32_t v423_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v430_data(0.0f);
              v430_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[51]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v432_data(0.0f);
              v432_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v432_data + (v384_data * v430_data)).copy_to(ir0 + (3));
              }
              int32_t v438_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v445_data(0.0f);
              v445_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[67]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v447_data(0.0f);
              v447_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v447_data + (v384_data * v445_data)).copy_to(ir0 + (4));
              }
              int32_t v453_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v460_data(0.0f);
              v460_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[83]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v462_data(0.0f);
              v462_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v462_data + (v384_data * v460_data)).copy_to(ir0 + (5));
              }
              int32_t v468_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v475_data(0.0f);
              v475_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[99]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v477_data(0.0f);
              v477_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v477_data + (v384_data * v475_data)).copy_to(ir0 + (6));
              }
              int32_t v483_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v490_data(0.0f);
              v490_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[115]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v492_data(0.0f);
              v492_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v492_data + (v384_data * v490_data)).copy_to(ir0 + (7));
              }
              int32_t v500_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v506_data(0.0f);
              v506_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[48_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v507_data(0.0f);
              v507_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[4]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v509_data(0.0f);
              v509_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v509_data + (v506_data * v507_data)).copy_to(ir0 + (0));
              }
              int32_t v515_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v522_data(0.0f);
              v522_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[20]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v524_data(0.0f);
              v524_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v524_data + (v506_data * v522_data)).copy_to(ir0 + (1));
              }
              int32_t v530_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v537_data(0.0f);
              v537_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[36]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v539_data(0.0f);
              v539_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v539_data + (v506_data * v537_data)).copy_to(ir0 + (2));
              }
              int32_t v545_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v552_data(0.0f);
              v552_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[52]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v554_data(0.0f);
              v554_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v554_data + (v506_data * v552_data)).copy_to(ir0 + (3));
              }
              int32_t v560_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v567_data(0.0f);
              v567_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[68]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v569_data(0.0f);
              v569_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v569_data + (v506_data * v567_data)).copy_to(ir0 + (4));
              }
              int32_t v575_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v582_data(0.0f);
              v582_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[84]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v584_data(0.0f);
              v584_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v584_data + (v506_data * v582_data)).copy_to(ir0 + (5));
              }
              int32_t v590_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v597_data(0.0f);
              v597_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[100]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v599_data(0.0f);
              v599_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v599_data + (v506_data * v597_data)).copy_to(ir0 + (6));
              }
              int32_t v605_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v612_data(0.0f);
              v612_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[116]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v614_data(0.0f);
              v614_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v614_data + (v506_data * v612_data)).copy_to(ir0 + (7));
              }
              int32_t v622_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v628_data(0.0f);
              v628_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[60_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v629_data(0.0f);
              v629_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[5]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v631_data(0.0f);
              v631_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v631_data + (v628_data * v629_data)).copy_to(ir0 + (0));
              }
              int32_t v637_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v644_data(0.0f);
              v644_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[21]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v646_data(0.0f);
              v646_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v646_data + (v628_data * v644_data)).copy_to(ir0 + (1));
              }
              int32_t v652_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v659_data(0.0f);
              v659_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[37]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v661_data(0.0f);
              v661_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v661_data + (v628_data * v659_data)).copy_to(ir0 + (2));
              }
              int32_t v667_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v674_data(0.0f);
              v674_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[53]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v676_data(0.0f);
              v676_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v676_data + (v628_data * v674_data)).copy_to(ir0 + (3));
              }
              int32_t v682_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v689_data(0.0f);
              v689_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[69]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v691_data(0.0f);
              v691_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v691_data + (v628_data * v689_data)).copy_to(ir0 + (4));
              }
              int32_t v697_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v704_data(0.0f);
              v704_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[85]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v706_data(0.0f);
              v706_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v706_data + (v628_data * v704_data)).copy_to(ir0 + (5));
              }
              int32_t v712_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v719_data(0.0f);
              v719_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[101]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v721_data(0.0f);
              v721_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v721_data + (v628_data * v719_data)).copy_to(ir0 + (6));
              }
              int32_t v727_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v734_data(0.0f);
              v734_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[117]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v736_data(0.0f);
              v736_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v736_data + (v628_data * v734_data)).copy_to(ir0 + (7));
              }
              int32_t v744_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v750_data(0.0f);
              v750_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[72_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v751_data(0.0f);
              v751_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[6]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v753_data(0.0f);
              v753_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v753_data + (v750_data * v751_data)).copy_to(ir0 + (0));
              }
              int32_t v759_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v766_data(0.0f);
              v766_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[22]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v768_data(0.0f);
              v768_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v768_data + (v750_data * v766_data)).copy_to(ir0 + (1));
              }
              int32_t v774_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v781_data(0.0f);
              v781_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[38]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v783_data(0.0f);
              v783_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v783_data + (v750_data * v781_data)).copy_to(ir0 + (2));
              }
              int32_t v789_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v796_data(0.0f);
              v796_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[54]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v798_data(0.0f);
              v798_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v798_data + (v750_data * v796_data)).copy_to(ir0 + (3));
              }
              int32_t v804_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v811_data(0.0f);
              v811_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[70]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v813_data(0.0f);
              v813_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v813_data + (v750_data * v811_data)).copy_to(ir0 + (4));
              }
              int32_t v819_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v826_data(0.0f);
              v826_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[86]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v828_data(0.0f);
              v828_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v828_data + (v750_data * v826_data)).copy_to(ir0 + (5));
              }
              int32_t v834_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v841_data(0.0f);
              v841_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[102]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v843_data(0.0f);
              v843_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v843_data + (v750_data * v841_data)).copy_to(ir0 + (6));
              }
              int32_t v849_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v856_data(0.0f);
              v856_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[118]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v858_data(0.0f);
              v858_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v858_data + (v750_data * v856_data)).copy_to(ir0 + (7));
              }
              int32_t v866_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v872_data(0.0f);
              v872_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[84_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v873_data(0.0f);
              v873_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[7]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v875_data(0.0f);
              v875_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v875_data + (v872_data * v873_data)).copy_to(ir0 + (0));
              }
              int32_t v881_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v888_data(0.0f);
              v888_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[23]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v890_data(0.0f);
              v890_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v890_data + (v872_data * v888_data)).copy_to(ir0 + (1));
              }
              int32_t v896_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v903_data(0.0f);
              v903_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[39]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v905_data(0.0f);
              v905_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v905_data + (v872_data * v903_data)).copy_to(ir0 + (2));
              }
              int32_t v911_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v918_data(0.0f);
              v918_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[55]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v920_data(0.0f);
              v920_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v920_data + (v872_data * v918_data)).copy_to(ir0 + (3));
              }
              int32_t v926_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v933_data(0.0f);
              v933_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[71]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v935_data(0.0f);
              v935_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v935_data + (v872_data * v933_data)).copy_to(ir0 + (4));
              }
              int32_t v941_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v948_data(0.0f);
              v948_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[87]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v950_data(0.0f);
              v950_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v950_data + (v872_data * v948_data)).copy_to(ir0 + (5));
              }
              int32_t v956_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v963_data(0.0f);
              v963_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[103]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v965_data(0.0f);
              v965_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v965_data + (v872_data * v963_data)).copy_to(ir0 + (6));
              }
              int32_t v971_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v978_data(0.0f);
              v978_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[119]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v980_data(0.0f);
              v980_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v980_data + (v872_data * v978_data)).copy_to(ir0 + (7));
              }
              int32_t v988_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v994_data(0.0f);
              v994_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[96_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v995_data(0.0f);
              v995_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[8]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v997_data(0.0f);
              v997_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v997_data + (v994_data * v995_data)).copy_to(ir0 + (0));
              }
              int32_t v1003_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1010_data(0.0f);
              v1010_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[24]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1012_data(0.0f);
              v1012_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1012_data + (v994_data * v1010_data)).copy_to(ir0 + (1));
              }
              int32_t v1018_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1025_data(0.0f);
              v1025_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[40]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1027_data(0.0f);
              v1027_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1027_data + (v994_data * v1025_data)).copy_to(ir0 + (2));
              }
              int32_t v1033_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1040_data(0.0f);
              v1040_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[56]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1042_data(0.0f);
              v1042_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1042_data + (v994_data * v1040_data)).copy_to(ir0 + (3));
              }
              int32_t v1048_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1055_data(0.0f);
              v1055_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[72]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1057_data(0.0f);
              v1057_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1057_data + (v994_data * v1055_data)).copy_to(ir0 + (4));
              }
              int32_t v1063_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1070_data(0.0f);
              v1070_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[88]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1072_data(0.0f);
              v1072_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1072_data + (v994_data * v1070_data)).copy_to(ir0 + (5));
              }
              int32_t v1078_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1085_data(0.0f);
              v1085_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[104]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1087_data(0.0f);
              v1087_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1087_data + (v994_data * v1085_data)).copy_to(ir0 + (6));
              }
              int32_t v1093_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1100_data(0.0f);
              v1100_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[120]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1102_data(0.0f);
              v1102_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1102_data + (v994_data * v1100_data)).copy_to(ir0 + (7));
              }
              int32_t v1110_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1116_data(0.0f);
              v1116_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[108_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1117_data(0.0f);
              v1117_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[9]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1119_data(0.0f);
              v1119_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1119_data + (v1116_data * v1117_data)).copy_to(ir0 + (0));
              }
              int32_t v1125_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1132_data(0.0f);
              v1132_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[25]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1134_data(0.0f);
              v1134_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1134_data + (v1116_data * v1132_data)).copy_to(ir0 + (1));
              }
              int32_t v1140_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1147_data(0.0f);
              v1147_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[41]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1149_data(0.0f);
              v1149_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1149_data + (v1116_data * v1147_data)).copy_to(ir0 + (2));
              }
              int32_t v1155_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1162_data(0.0f);
              v1162_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[57]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1164_data(0.0f);
              v1164_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1164_data + (v1116_data * v1162_data)).copy_to(ir0 + (3));
              }
              int32_t v1170_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1177_data(0.0f);
              v1177_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[73]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1179_data(0.0f);
              v1179_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1179_data + (v1116_data * v1177_data)).copy_to(ir0 + (4));
              }
              int32_t v1185_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1192_data(0.0f);
              v1192_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[89]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1194_data(0.0f);
              v1194_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1194_data + (v1116_data * v1192_data)).copy_to(ir0 + (5));
              }
              int32_t v1200_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1207_data(0.0f);
              v1207_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[105]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1209_data(0.0f);
              v1209_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1209_data + (v1116_data * v1207_data)).copy_to(ir0 + (6));
              }
              int32_t v1215_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1222_data(0.0f);
              v1222_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[121]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1224_data(0.0f);
              v1224_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1224_data + (v1116_data * v1222_data)).copy_to(ir0 + (7));
              }
              int32_t v1232_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1238_data(0.0f);
              v1238_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[120_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1239_data(0.0f);
              v1239_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[10]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1241_data(0.0f);
              v1241_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1241_data + (v1238_data * v1239_data)).copy_to(ir0 + (0));
              }
              int32_t v1247_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1254_data(0.0f);
              v1254_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[26]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1256_data(0.0f);
              v1256_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1256_data + (v1238_data * v1254_data)).copy_to(ir0 + (1));
              }
              int32_t v1262_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1269_data(0.0f);
              v1269_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[42]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1271_data(0.0f);
              v1271_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1271_data + (v1238_data * v1269_data)).copy_to(ir0 + (2));
              }
              int32_t v1277_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1284_data(0.0f);
              v1284_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[58]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1286_data(0.0f);
              v1286_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1286_data + (v1238_data * v1284_data)).copy_to(ir0 + (3));
              }
              int32_t v1292_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1299_data(0.0f);
              v1299_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[74]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1301_data(0.0f);
              v1301_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1301_data + (v1238_data * v1299_data)).copy_to(ir0 + (4));
              }
              int32_t v1307_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1314_data(0.0f);
              v1314_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[90]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1316_data(0.0f);
              v1316_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1316_data + (v1238_data * v1314_data)).copy_to(ir0 + (5));
              }
              int32_t v1322_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1329_data(0.0f);
              v1329_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[106]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1331_data(0.0f);
              v1331_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1331_data + (v1238_data * v1329_data)).copy_to(ir0 + (6));
              }
              int32_t v1337_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1344_data(0.0f);
              v1344_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[122]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1346_data(0.0f);
              v1346_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1346_data + (v1238_data * v1344_data)).copy_to(ir0 + (7));
              }
              int32_t v1354_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1360_data(0.0f);
              v1360_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[132_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1361_data(0.0f);
              v1361_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[11]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1363_data(0.0f);
              v1363_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1363_data + (v1360_data * v1361_data)).copy_to(ir0 + (0));
              }
              int32_t v1369_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1376_data(0.0f);
              v1376_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[27]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1378_data(0.0f);
              v1378_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1378_data + (v1360_data * v1376_data)).copy_to(ir0 + (1));
              }
              int32_t v1384_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1391_data(0.0f);
              v1391_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[43]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1393_data(0.0f);
              v1393_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1393_data + (v1360_data * v1391_data)).copy_to(ir0 + (2));
              }
              int32_t v1399_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1406_data(0.0f);
              v1406_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[59]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1408_data(0.0f);
              v1408_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1408_data + (v1360_data * v1406_data)).copy_to(ir0 + (3));
              }
              int32_t v1414_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1421_data(0.0f);
              v1421_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[75]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1423_data(0.0f);
              v1423_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1423_data + (v1360_data * v1421_data)).copy_to(ir0 + (4));
              }
              int32_t v1429_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1436_data(0.0f);
              v1436_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[91]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1438_data(0.0f);
              v1438_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1438_data + (v1360_data * v1436_data)).copy_to(ir0 + (5));
              }
              int32_t v1444_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1451_data(0.0f);
              v1451_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[107]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1453_data(0.0f);
              v1453_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1453_data + (v1360_data * v1451_data)).copy_to(ir0 + (6));
              }
              int32_t v1459_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1466_data(0.0f);
              v1466_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[123]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1468_data(0.0f);
              v1468_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1468_data + (v1360_data * v1466_data)).copy_to(ir0 + (7));
              }
              int32_t v1476_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1482_data(0.0f);
              v1482_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[144_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1483_data(0.0f);
              v1483_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[12]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1485_data(0.0f);
              v1485_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1485_data + (v1482_data * v1483_data)).copy_to(ir0 + (0));
              }
              int32_t v1491_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1498_data(0.0f);
              v1498_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[28]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1500_data(0.0f);
              v1500_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1500_data + (v1482_data * v1498_data)).copy_to(ir0 + (1));
              }
              int32_t v1506_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1513_data(0.0f);
              v1513_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[44]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1515_data(0.0f);
              v1515_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1515_data + (v1482_data * v1513_data)).copy_to(ir0 + (2));
              }
              int32_t v1521_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1528_data(0.0f);
              v1528_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[60]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1530_data(0.0f);
              v1530_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1530_data + (v1482_data * v1528_data)).copy_to(ir0 + (3));
              }
              int32_t v1536_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1543_data(0.0f);
              v1543_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[76]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1545_data(0.0f);
              v1545_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1545_data + (v1482_data * v1543_data)).copy_to(ir0 + (4));
              }
              int32_t v1551_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1558_data(0.0f);
              v1558_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[92]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1560_data(0.0f);
              v1560_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1560_data + (v1482_data * v1558_data)).copy_to(ir0 + (5));
              }
              int32_t v1566_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1573_data(0.0f);
              v1573_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[108]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1575_data(0.0f);
              v1575_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1575_data + (v1482_data * v1573_data)).copy_to(ir0 + (6));
              }
              int32_t v1581_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1588_data(0.0f);
              v1588_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[124]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1590_data(0.0f);
              v1590_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1590_data + (v1482_data * v1588_data)).copy_to(ir0 + (7));
              }
              int32_t v1598_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1604_data(0.0f);
              v1604_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[156_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1605_data(0.0f);
              v1605_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[13]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1607_data(0.0f);
              v1607_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1607_data + (v1604_data * v1605_data)).copy_to(ir0 + (0));
              }
              int32_t v1613_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1620_data(0.0f);
              v1620_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[29]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1622_data(0.0f);
              v1622_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1622_data + (v1604_data * v1620_data)).copy_to(ir0 + (1));
              }
              int32_t v1628_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1635_data(0.0f);
              v1635_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[45]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1637_data(0.0f);
              v1637_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1637_data + (v1604_data * v1635_data)).copy_to(ir0 + (2));
              }
              int32_t v1643_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1650_data(0.0f);
              v1650_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[61]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1652_data(0.0f);
              v1652_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1652_data + (v1604_data * v1650_data)).copy_to(ir0 + (3));
              }
              int32_t v1658_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1665_data(0.0f);
              v1665_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[77]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1667_data(0.0f);
              v1667_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1667_data + (v1604_data * v1665_data)).copy_to(ir0 + (4));
              }
              int32_t v1673_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1680_data(0.0f);
              v1680_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[93]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1682_data(0.0f);
              v1682_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1682_data + (v1604_data * v1680_data)).copy_to(ir0 + (5));
              }
              int32_t v1688_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1695_data(0.0f);
              v1695_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[109]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1697_data(0.0f);
              v1697_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1697_data + (v1604_data * v1695_data)).copy_to(ir0 + (6));
              }
              int32_t v1703_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1710_data(0.0f);
              v1710_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[125]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1712_data(0.0f);
              v1712_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1712_data + (v1604_data * v1710_data)).copy_to(ir0 + (7));
              }
              int32_t v1720_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1726_data(0.0f);
              v1726_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[168_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1727_data(0.0f);
              v1727_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[14]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1729_data(0.0f);
              v1729_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1729_data + (v1726_data * v1727_data)).copy_to(ir0 + (0));
              }
              int32_t v1735_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1742_data(0.0f);
              v1742_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[30]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1744_data(0.0f);
              v1744_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1744_data + (v1726_data * v1742_data)).copy_to(ir0 + (1));
              }
              int32_t v1750_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1757_data(0.0f);
              v1757_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[46]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1759_data(0.0f);
              v1759_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1759_data + (v1726_data * v1757_data)).copy_to(ir0 + (2));
              }
              int32_t v1765_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1772_data(0.0f);
              v1772_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[62]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1774_data(0.0f);
              v1774_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1774_data + (v1726_data * v1772_data)).copy_to(ir0 + (3));
              }
              int32_t v1780_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1787_data(0.0f);
              v1787_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[78]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1789_data(0.0f);
              v1789_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1789_data + (v1726_data * v1787_data)).copy_to(ir0 + (4));
              }
              int32_t v1795_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1802_data(0.0f);
              v1802_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[94]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1804_data(0.0f);
              v1804_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1804_data + (v1726_data * v1802_data)).copy_to(ir0 + (5));
              }
              int32_t v1810_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1817_data(0.0f);
              v1817_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[110]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1819_data(0.0f);
              v1819_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1819_data + (v1726_data * v1817_data)).copy_to(ir0 + (6));
              }
              int32_t v1825_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1832_data(0.0f);
              v1832_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[126]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1834_data(0.0f);
              v1834_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1834_data + (v1726_data * v1832_data)).copy_to(ir0 + (7));
              }
              int32_t v1842_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1848_data(0.0f);
              v1848_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[180_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1849_data(0.0f);
              v1849_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[15]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1851_data(0.0f);
              v1851_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1851_data + (v1848_data * v1849_data)).copy_to(ir0 + (0));
              }
              int32_t v1857_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1864_data(0.0f);
              v1864_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[31]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1866_data(0.0f);
              v1866_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1866_data + (v1848_data * v1864_data)).copy_to(ir0 + (1));
              }
              int32_t v1872_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1879_data(0.0f);
              v1879_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[47]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1881_data(0.0f);
              v1881_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1881_data + (v1848_data * v1879_data)).copy_to(ir0 + (2));
              }
              int32_t v1887_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1894_data(0.0f);
              v1894_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[63]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1896_data(0.0f);
              v1896_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1896_data + (v1848_data * v1894_data)).copy_to(ir0 + (3));
              }
              int32_t v1902_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1909_data(0.0f);
              v1909_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[79]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1911_data(0.0f);
              v1911_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1911_data + (v1848_data * v1909_data)).copy_to(ir0 + (4));
              }
              int32_t v1917_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1924_data(0.0f);
              v1924_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[95]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1926_data(0.0f);
              v1926_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1926_data + (v1848_data * v1924_data)).copy_to(ir0 + (5));
              }
              int32_t v1932_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1939_data(0.0f);
              v1939_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[111]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1941_data(0.0f);
              v1941_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1941_data + (v1848_data * v1939_data)).copy_to(ir0 + (6));
              }
              int32_t v1947_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1954_data(0.0f);
              v1954_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[127]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1956_data(0.0f);
              v1956_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1956_data + (v1848_data * v1954_data)).copy_to(ir0 + (7));
              }
              #pragma unroll
              for (int32_t v1960_n1 = 0; v1960_n1 < 8; ++v1960_n1) {
                int32_t v1961_a = 0 + v1960_n1;
                tensorforge::intel_esimd::simd<float, 16> v1963_data(0.0f);
                v1963_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[v1960_n1]), v7_g);
                if (v7_g) {
                  v1963_data.copy_to(r0 + (v1960_n1));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v1967_i1 = 0; v1967_i1 < 8; ++v1967_i1) {
                int32_t v1968_a = 0 + v1967_i1;
                tensorforge::intel_esimd::simd<float, 16> v1970_data(0.0f);
                v1970_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[v1967_i1]), v7_g);
                if (v7_g) {
                  v1970_data.copy_to(glb_m0 + ((v1967_i1 * 12)));
                }
              }
            }
          }
        }
      });
    }
  });
}

