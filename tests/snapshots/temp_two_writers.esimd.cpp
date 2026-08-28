// === base name ===
kernel_3e24e7feaf

// === header ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_3e24e7feaf(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3e24e7feaf(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3e24e7feaf(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2560, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32×32(6×12) {0..6}×{0..12} strided
        // m1 32×32(12×12) {0..12}×{0..12} strided
        // m2 32×32(6×12) {0..6}×{0..12} strided
        // m3 32×32(12×12) {0..12}×{0..12} strided
        // m4 32×32(12×12) {0..12}×{0..12} strided
        // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m0 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
        // t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..6}×{0..12})[0, 1] = m2 32×32(6×12) {0..6}×{0..12} strided({0..6}×{0..12})[0, -1]×m1 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[-1, 1]
        // m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, 1] = m4 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×12(12×12) {0..12}×{0..12} pointer_based({0..12}×{0..12})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[160 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[144];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 144 + 0 + m4_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m1[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m1[0 + 0 + 4 * item.get_local_id(0) + 64];
              s0[0 + 0 + 1 * item.get_local_id(0) + 128] = glb_m1[0 + 0 + 1 * item.get_local_id(0) + 128];
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r0[12]{};
              // r0 = +(glb_m0 * s0) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<int32_t, 16> v7_lead = tensorforge::intel_esimd::simd<int32_t, 16>(0, 1);
              tensorforge::intel_esimd::simd_mask<16> v8_g = v7_lead < 6;
              int32_t v11_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v15_data(0.0f);
              v15_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[0_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v16_data(0.0f);
              v16_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[0]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v18_data(0.0f);
              v18_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v8_g);
              if (v8_g) {
                (v18_data + (v15_data * v16_data)).copy_to(r0 + (0));
              }
              int32_t v22_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v27_data(0.0f);
              v27_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[12]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v29_data(0.0f);
              v29_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v8_g);
              if (v8_g) {
                (v29_data + (v15_data * v27_data)).copy_to(r0 + (1));
              }
              int32_t v33_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v38_data(0.0f);
              v38_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[24]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v40_data(0.0f);
              v40_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v8_g);
              if (v8_g) {
                (v40_data + (v15_data * v38_data)).copy_to(r0 + (2));
              }
              int32_t v44_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v49_data(0.0f);
              v49_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[36]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v51_data(0.0f);
              v51_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v8_g);
              if (v8_g) {
                (v51_data + (v15_data * v49_data)).copy_to(r0 + (3));
              }
              int32_t v55_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v60_data(0.0f);
              v60_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[48]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v62_data(0.0f);
              v62_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v8_g);
              if (v8_g) {
                (v62_data + (v15_data * v60_data)).copy_to(r0 + (4));
              }
              int32_t v66_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v71_data(0.0f);
              v71_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[60]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v73_data(0.0f);
              v73_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v8_g);
              if (v8_g) {
                (v73_data + (v15_data * v71_data)).copy_to(r0 + (5));
              }
              int32_t v77_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v82_data(0.0f);
              v82_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[72]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v84_data(0.0f);
              v84_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[6]), v8_g);
              if (v8_g) {
                (v84_data + (v15_data * v82_data)).copy_to(r0 + (6));
              }
              int32_t v88_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v93_data(0.0f);
              v93_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[84]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v95_data(0.0f);
              v95_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[7]), v8_g);
              if (v8_g) {
                (v95_data + (v15_data * v93_data)).copy_to(r0 + (7));
              }
              int32_t v99_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v104_data(0.0f);
              v104_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[96]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v106_data(0.0f);
              v106_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[8]), v8_g);
              if (v8_g) {
                (v106_data + (v15_data * v104_data)).copy_to(r0 + (8));
              }
              int32_t v110_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v115_data(0.0f);
              v115_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[108]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v117_data(0.0f);
              v117_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[9]), v8_g);
              if (v8_g) {
                (v117_data + (v15_data * v115_data)).copy_to(r0 + (9));
              }
              int32_t v121_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v126_data(0.0f);
              v126_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[120]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v128_data(0.0f);
              v128_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[10]), v8_g);
              if (v8_g) {
                (v128_data + (v15_data * v126_data)).copy_to(r0 + (10));
              }
              int32_t v132_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v137_data(0.0f);
              v137_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[132]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v139_data(0.0f);
              v139_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[11]), v8_g);
              if (v8_g) {
                (v139_data + (v15_data * v137_data)).copy_to(r0 + (11));
              }
              int32_t v145_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v149_data(0.0f);
              v149_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[6_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v150_data(0.0f);
              v150_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[1]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v152_data(0.0f);
              v152_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v8_g);
              if (v8_g) {
                (v152_data + (v149_data * v150_data)).copy_to(r0 + (0));
              }
              int32_t v156_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v161_data(0.0f);
              v161_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[13]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v163_data(0.0f);
              v163_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v8_g);
              if (v8_g) {
                (v163_data + (v149_data * v161_data)).copy_to(r0 + (1));
              }
              int32_t v167_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v172_data(0.0f);
              v172_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[25]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v174_data(0.0f);
              v174_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v8_g);
              if (v8_g) {
                (v174_data + (v149_data * v172_data)).copy_to(r0 + (2));
              }
              int32_t v178_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v183_data(0.0f);
              v183_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[37]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v185_data(0.0f);
              v185_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v8_g);
              if (v8_g) {
                (v185_data + (v149_data * v183_data)).copy_to(r0 + (3));
              }
              int32_t v189_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v194_data(0.0f);
              v194_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[49]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v196_data(0.0f);
              v196_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v8_g);
              if (v8_g) {
                (v196_data + (v149_data * v194_data)).copy_to(r0 + (4));
              }
              int32_t v200_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v205_data(0.0f);
              v205_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[61]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v207_data(0.0f);
              v207_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v8_g);
              if (v8_g) {
                (v207_data + (v149_data * v205_data)).copy_to(r0 + (5));
              }
              int32_t v211_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v216_data(0.0f);
              v216_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[73]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v218_data(0.0f);
              v218_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[6]), v8_g);
              if (v8_g) {
                (v218_data + (v149_data * v216_data)).copy_to(r0 + (6));
              }
              int32_t v222_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v227_data(0.0f);
              v227_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[85]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v229_data(0.0f);
              v229_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[7]), v8_g);
              if (v8_g) {
                (v229_data + (v149_data * v227_data)).copy_to(r0 + (7));
              }
              int32_t v233_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v238_data(0.0f);
              v238_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[97]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v240_data(0.0f);
              v240_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[8]), v8_g);
              if (v8_g) {
                (v240_data + (v149_data * v238_data)).copy_to(r0 + (8));
              }
              int32_t v244_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v249_data(0.0f);
              v249_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[109]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v251_data(0.0f);
              v251_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[9]), v8_g);
              if (v8_g) {
                (v251_data + (v149_data * v249_data)).copy_to(r0 + (9));
              }
              int32_t v255_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v260_data(0.0f);
              v260_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[121]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v262_data(0.0f);
              v262_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[10]), v8_g);
              if (v8_g) {
                (v262_data + (v149_data * v260_data)).copy_to(r0 + (10));
              }
              int32_t v266_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v271_data(0.0f);
              v271_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[133]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v273_data(0.0f);
              v273_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[11]), v8_g);
              if (v8_g) {
                (v273_data + (v149_data * v271_data)).copy_to(r0 + (11));
              }
              int32_t v279_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v283_data(0.0f);
              v283_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[12_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v284_data(0.0f);
              v284_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[2]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v286_data(0.0f);
              v286_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v8_g);
              if (v8_g) {
                (v286_data + (v283_data * v284_data)).copy_to(r0 + (0));
              }
              int32_t v290_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v295_data(0.0f);
              v295_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[14]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v297_data(0.0f);
              v297_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v8_g);
              if (v8_g) {
                (v297_data + (v283_data * v295_data)).copy_to(r0 + (1));
              }
              int32_t v301_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v306_data(0.0f);
              v306_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[26]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v308_data(0.0f);
              v308_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v8_g);
              if (v8_g) {
                (v308_data + (v283_data * v306_data)).copy_to(r0 + (2));
              }
              int32_t v312_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v317_data(0.0f);
              v317_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[38]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v319_data(0.0f);
              v319_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v8_g);
              if (v8_g) {
                (v319_data + (v283_data * v317_data)).copy_to(r0 + (3));
              }
              int32_t v323_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v328_data(0.0f);
              v328_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[50]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v330_data(0.0f);
              v330_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v8_g);
              if (v8_g) {
                (v330_data + (v283_data * v328_data)).copy_to(r0 + (4));
              }
              int32_t v334_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v339_data(0.0f);
              v339_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[62]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v341_data(0.0f);
              v341_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v8_g);
              if (v8_g) {
                (v341_data + (v283_data * v339_data)).copy_to(r0 + (5));
              }
              int32_t v345_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v350_data(0.0f);
              v350_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[74]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v352_data(0.0f);
              v352_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[6]), v8_g);
              if (v8_g) {
                (v352_data + (v283_data * v350_data)).copy_to(r0 + (6));
              }
              int32_t v356_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v361_data(0.0f);
              v361_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[86]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v363_data(0.0f);
              v363_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[7]), v8_g);
              if (v8_g) {
                (v363_data + (v283_data * v361_data)).copy_to(r0 + (7));
              }
              int32_t v367_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v372_data(0.0f);
              v372_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[98]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v374_data(0.0f);
              v374_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[8]), v8_g);
              if (v8_g) {
                (v374_data + (v283_data * v372_data)).copy_to(r0 + (8));
              }
              int32_t v378_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v383_data(0.0f);
              v383_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[110]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v385_data(0.0f);
              v385_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[9]), v8_g);
              if (v8_g) {
                (v385_data + (v283_data * v383_data)).copy_to(r0 + (9));
              }
              int32_t v389_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v394_data(0.0f);
              v394_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[122]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v396_data(0.0f);
              v396_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[10]), v8_g);
              if (v8_g) {
                (v396_data + (v283_data * v394_data)).copy_to(r0 + (10));
              }
              int32_t v400_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v405_data(0.0f);
              v405_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[134]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v407_data(0.0f);
              v407_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[11]), v8_g);
              if (v8_g) {
                (v407_data + (v283_data * v405_data)).copy_to(r0 + (11));
              }
              int32_t v413_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v417_data(0.0f);
              v417_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[18_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v418_data(0.0f);
              v418_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[3]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v420_data(0.0f);
              v420_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v8_g);
              if (v8_g) {
                (v420_data + (v417_data * v418_data)).copy_to(r0 + (0));
              }
              int32_t v424_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v429_data(0.0f);
              v429_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[15]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v431_data(0.0f);
              v431_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v8_g);
              if (v8_g) {
                (v431_data + (v417_data * v429_data)).copy_to(r0 + (1));
              }
              int32_t v435_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v440_data(0.0f);
              v440_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[27]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v442_data(0.0f);
              v442_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v8_g);
              if (v8_g) {
                (v442_data + (v417_data * v440_data)).copy_to(r0 + (2));
              }
              int32_t v446_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v451_data(0.0f);
              v451_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[39]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v453_data(0.0f);
              v453_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v8_g);
              if (v8_g) {
                (v453_data + (v417_data * v451_data)).copy_to(r0 + (3));
              }
              int32_t v457_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v462_data(0.0f);
              v462_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[51]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v464_data(0.0f);
              v464_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v8_g);
              if (v8_g) {
                (v464_data + (v417_data * v462_data)).copy_to(r0 + (4));
              }
              int32_t v468_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v473_data(0.0f);
              v473_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[63]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v475_data(0.0f);
              v475_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v8_g);
              if (v8_g) {
                (v475_data + (v417_data * v473_data)).copy_to(r0 + (5));
              }
              int32_t v479_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v484_data(0.0f);
              v484_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[75]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v486_data(0.0f);
              v486_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[6]), v8_g);
              if (v8_g) {
                (v486_data + (v417_data * v484_data)).copy_to(r0 + (6));
              }
              int32_t v490_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v495_data(0.0f);
              v495_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[87]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v497_data(0.0f);
              v497_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[7]), v8_g);
              if (v8_g) {
                (v497_data + (v417_data * v495_data)).copy_to(r0 + (7));
              }
              int32_t v501_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v506_data(0.0f);
              v506_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[99]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v508_data(0.0f);
              v508_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[8]), v8_g);
              if (v8_g) {
                (v508_data + (v417_data * v506_data)).copy_to(r0 + (8));
              }
              int32_t v512_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v517_data(0.0f);
              v517_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[111]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v519_data(0.0f);
              v519_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[9]), v8_g);
              if (v8_g) {
                (v519_data + (v417_data * v517_data)).copy_to(r0 + (9));
              }
              int32_t v523_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v528_data(0.0f);
              v528_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[123]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v530_data(0.0f);
              v530_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[10]), v8_g);
              if (v8_g) {
                (v530_data + (v417_data * v528_data)).copy_to(r0 + (10));
              }
              int32_t v534_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v539_data(0.0f);
              v539_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[135]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v541_data(0.0f);
              v541_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[11]), v8_g);
              if (v8_g) {
                (v541_data + (v417_data * v539_data)).copy_to(r0 + (11));
              }
              int32_t v547_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v551_data(0.0f);
              v551_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[24_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v552_data(0.0f);
              v552_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[4]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v554_data(0.0f);
              v554_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v8_g);
              if (v8_g) {
                (v554_data + (v551_data * v552_data)).copy_to(r0 + (0));
              }
              int32_t v558_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v563_data(0.0f);
              v563_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[16]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v565_data(0.0f);
              v565_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v8_g);
              if (v8_g) {
                (v565_data + (v551_data * v563_data)).copy_to(r0 + (1));
              }
              int32_t v569_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v574_data(0.0f);
              v574_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[28]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v576_data(0.0f);
              v576_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v8_g);
              if (v8_g) {
                (v576_data + (v551_data * v574_data)).copy_to(r0 + (2));
              }
              int32_t v580_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v585_data(0.0f);
              v585_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[40]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v587_data(0.0f);
              v587_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v8_g);
              if (v8_g) {
                (v587_data + (v551_data * v585_data)).copy_to(r0 + (3));
              }
              int32_t v591_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v596_data(0.0f);
              v596_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[52]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v598_data(0.0f);
              v598_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v8_g);
              if (v8_g) {
                (v598_data + (v551_data * v596_data)).copy_to(r0 + (4));
              }
              int32_t v602_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v607_data(0.0f);
              v607_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[64]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v609_data(0.0f);
              v609_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v8_g);
              if (v8_g) {
                (v609_data + (v551_data * v607_data)).copy_to(r0 + (5));
              }
              int32_t v613_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v618_data(0.0f);
              v618_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[76]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v620_data(0.0f);
              v620_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[6]), v8_g);
              if (v8_g) {
                (v620_data + (v551_data * v618_data)).copy_to(r0 + (6));
              }
              int32_t v624_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v629_data(0.0f);
              v629_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[88]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v631_data(0.0f);
              v631_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[7]), v8_g);
              if (v8_g) {
                (v631_data + (v551_data * v629_data)).copy_to(r0 + (7));
              }
              int32_t v635_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v640_data(0.0f);
              v640_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[100]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v642_data(0.0f);
              v642_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[8]), v8_g);
              if (v8_g) {
                (v642_data + (v551_data * v640_data)).copy_to(r0 + (8));
              }
              int32_t v646_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v651_data(0.0f);
              v651_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[112]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v653_data(0.0f);
              v653_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[9]), v8_g);
              if (v8_g) {
                (v653_data + (v551_data * v651_data)).copy_to(r0 + (9));
              }
              int32_t v657_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v662_data(0.0f);
              v662_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[124]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v664_data(0.0f);
              v664_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[10]), v8_g);
              if (v8_g) {
                (v664_data + (v551_data * v662_data)).copy_to(r0 + (10));
              }
              int32_t v668_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v673_data(0.0f);
              v673_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[136]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v675_data(0.0f);
              v675_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[11]), v8_g);
              if (v8_g) {
                (v675_data + (v551_data * v673_data)).copy_to(r0 + (11));
              }
              int32_t v681_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v685_data(0.0f);
              v685_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[30_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v686_data(0.0f);
              v686_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[5]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v688_data(0.0f);
              v688_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v8_g);
              if (v8_g) {
                (v688_data + (v685_data * v686_data)).copy_to(r0 + (0));
              }
              int32_t v692_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v697_data(0.0f);
              v697_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[17]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v699_data(0.0f);
              v699_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v8_g);
              if (v8_g) {
                (v699_data + (v685_data * v697_data)).copy_to(r0 + (1));
              }
              int32_t v703_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v708_data(0.0f);
              v708_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[29]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v710_data(0.0f);
              v710_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v8_g);
              if (v8_g) {
                (v710_data + (v685_data * v708_data)).copy_to(r0 + (2));
              }
              int32_t v714_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v719_data(0.0f);
              v719_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[41]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v721_data(0.0f);
              v721_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v8_g);
              if (v8_g) {
                (v721_data + (v685_data * v719_data)).copy_to(r0 + (3));
              }
              int32_t v725_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v730_data(0.0f);
              v730_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[53]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v732_data(0.0f);
              v732_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v8_g);
              if (v8_g) {
                (v732_data + (v685_data * v730_data)).copy_to(r0 + (4));
              }
              int32_t v736_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v741_data(0.0f);
              v741_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[65]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v743_data(0.0f);
              v743_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v8_g);
              if (v8_g) {
                (v743_data + (v685_data * v741_data)).copy_to(r0 + (5));
              }
              int32_t v747_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v752_data(0.0f);
              v752_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[77]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v754_data(0.0f);
              v754_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[6]), v8_g);
              if (v8_g) {
                (v754_data + (v685_data * v752_data)).copy_to(r0 + (6));
              }
              int32_t v758_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v763_data(0.0f);
              v763_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[89]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v765_data(0.0f);
              v765_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[7]), v8_g);
              if (v8_g) {
                (v765_data + (v685_data * v763_data)).copy_to(r0 + (7));
              }
              int32_t v769_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v774_data(0.0f);
              v774_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[101]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v776_data(0.0f);
              v776_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[8]), v8_g);
              if (v8_g) {
                (v776_data + (v685_data * v774_data)).copy_to(r0 + (8));
              }
              int32_t v780_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v785_data(0.0f);
              v785_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[113]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v787_data(0.0f);
              v787_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[9]), v8_g);
              if (v8_g) {
                (v787_data + (v685_data * v785_data)).copy_to(r0 + (9));
              }
              int32_t v791_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v796_data(0.0f);
              v796_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[125]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v798_data(0.0f);
              v798_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[10]), v8_g);
              if (v8_g) {
                (v798_data + (v685_data * v796_data)).copy_to(r0 + (10));
              }
              int32_t v802_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v807_data(0.0f);
              v807_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[137]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v809_data(0.0f);
              v809_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[11]), v8_g);
              if (v8_g) {
                (v809_data + (v685_data * v807_data)).copy_to(r0 + (11));
              }
              int32_t v815_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v819_data(0.0f);
              v819_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[36_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v820_data(0.0f);
              v820_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[6]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v822_data(0.0f);
              v822_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v8_g);
              if (v8_g) {
                (v822_data + (v819_data * v820_data)).copy_to(r0 + (0));
              }
              int32_t v826_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v831_data(0.0f);
              v831_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[18]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v833_data(0.0f);
              v833_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v8_g);
              if (v8_g) {
                (v833_data + (v819_data * v831_data)).copy_to(r0 + (1));
              }
              int32_t v837_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v842_data(0.0f);
              v842_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[30]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v844_data(0.0f);
              v844_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v8_g);
              if (v8_g) {
                (v844_data + (v819_data * v842_data)).copy_to(r0 + (2));
              }
              int32_t v848_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v853_data(0.0f);
              v853_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[42]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v855_data(0.0f);
              v855_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v8_g);
              if (v8_g) {
                (v855_data + (v819_data * v853_data)).copy_to(r0 + (3));
              }
              int32_t v859_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v864_data(0.0f);
              v864_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[54]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v866_data(0.0f);
              v866_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v8_g);
              if (v8_g) {
                (v866_data + (v819_data * v864_data)).copy_to(r0 + (4));
              }
              int32_t v870_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v875_data(0.0f);
              v875_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[66]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v877_data(0.0f);
              v877_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v8_g);
              if (v8_g) {
                (v877_data + (v819_data * v875_data)).copy_to(r0 + (5));
              }
              int32_t v881_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v886_data(0.0f);
              v886_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[78]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v888_data(0.0f);
              v888_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[6]), v8_g);
              if (v8_g) {
                (v888_data + (v819_data * v886_data)).copy_to(r0 + (6));
              }
              int32_t v892_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v897_data(0.0f);
              v897_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[90]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v899_data(0.0f);
              v899_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[7]), v8_g);
              if (v8_g) {
                (v899_data + (v819_data * v897_data)).copy_to(r0 + (7));
              }
              int32_t v903_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v908_data(0.0f);
              v908_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[102]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v910_data(0.0f);
              v910_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[8]), v8_g);
              if (v8_g) {
                (v910_data + (v819_data * v908_data)).copy_to(r0 + (8));
              }
              int32_t v914_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v919_data(0.0f);
              v919_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[114]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v921_data(0.0f);
              v921_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[9]), v8_g);
              if (v8_g) {
                (v921_data + (v819_data * v919_data)).copy_to(r0 + (9));
              }
              int32_t v925_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v930_data(0.0f);
              v930_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[126]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v932_data(0.0f);
              v932_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[10]), v8_g);
              if (v8_g) {
                (v932_data + (v819_data * v930_data)).copy_to(r0 + (10));
              }
              int32_t v936_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v941_data(0.0f);
              v941_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[138]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v943_data(0.0f);
              v943_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[11]), v8_g);
              if (v8_g) {
                (v943_data + (v819_data * v941_data)).copy_to(r0 + (11));
              }
              int32_t v949_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v953_data(0.0f);
              v953_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[42_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v954_data(0.0f);
              v954_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[7]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v956_data(0.0f);
              v956_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v8_g);
              if (v8_g) {
                (v956_data + (v953_data * v954_data)).copy_to(r0 + (0));
              }
              int32_t v960_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v965_data(0.0f);
              v965_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[19]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v967_data(0.0f);
              v967_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v8_g);
              if (v8_g) {
                (v967_data + (v953_data * v965_data)).copy_to(r0 + (1));
              }
              int32_t v971_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v976_data(0.0f);
              v976_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[31]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v978_data(0.0f);
              v978_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v8_g);
              if (v8_g) {
                (v978_data + (v953_data * v976_data)).copy_to(r0 + (2));
              }
              int32_t v982_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v987_data(0.0f);
              v987_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[43]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v989_data(0.0f);
              v989_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v8_g);
              if (v8_g) {
                (v989_data + (v953_data * v987_data)).copy_to(r0 + (3));
              }
              int32_t v993_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v998_data(0.0f);
              v998_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[55]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1000_data(0.0f);
              v1000_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v8_g);
              if (v8_g) {
                (v1000_data + (v953_data * v998_data)).copy_to(r0 + (4));
              }
              int32_t v1004_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v1009_data(0.0f);
              v1009_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[67]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1011_data(0.0f);
              v1011_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v8_g);
              if (v8_g) {
                (v1011_data + (v953_data * v1009_data)).copy_to(r0 + (5));
              }
              int32_t v1015_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v1020_data(0.0f);
              v1020_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[79]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1022_data(0.0f);
              v1022_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[6]), v8_g);
              if (v8_g) {
                (v1022_data + (v953_data * v1020_data)).copy_to(r0 + (6));
              }
              int32_t v1026_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v1031_data(0.0f);
              v1031_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[91]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1033_data(0.0f);
              v1033_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[7]), v8_g);
              if (v8_g) {
                (v1033_data + (v953_data * v1031_data)).copy_to(r0 + (7));
              }
              int32_t v1037_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v1042_data(0.0f);
              v1042_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[103]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1044_data(0.0f);
              v1044_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[8]), v8_g);
              if (v8_g) {
                (v1044_data + (v953_data * v1042_data)).copy_to(r0 + (8));
              }
              int32_t v1048_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v1053_data(0.0f);
              v1053_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[115]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1055_data(0.0f);
              v1055_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[9]), v8_g);
              if (v8_g) {
                (v1055_data + (v953_data * v1053_data)).copy_to(r0 + (9));
              }
              int32_t v1059_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v1064_data(0.0f);
              v1064_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[127]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1066_data(0.0f);
              v1066_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[10]), v8_g);
              if (v8_g) {
                (v1066_data + (v953_data * v1064_data)).copy_to(r0 + (10));
              }
              int32_t v1070_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v1075_data(0.0f);
              v1075_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[139]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1077_data(0.0f);
              v1077_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[11]), v8_g);
              if (v8_g) {
                (v1077_data + (v953_data * v1075_data)).copy_to(r0 + (11));
              }
              int32_t v1083_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1087_data(0.0f);
              v1087_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[48_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1088_data(0.0f);
              v1088_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[8]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1090_data(0.0f);
              v1090_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v8_g);
              if (v8_g) {
                (v1090_data + (v1087_data * v1088_data)).copy_to(r0 + (0));
              }
              int32_t v1094_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1099_data(0.0f);
              v1099_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[20]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1101_data(0.0f);
              v1101_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v8_g);
              if (v8_g) {
                (v1101_data + (v1087_data * v1099_data)).copy_to(r0 + (1));
              }
              int32_t v1105_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1110_data(0.0f);
              v1110_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1112_data(0.0f);
              v1112_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v8_g);
              if (v8_g) {
                (v1112_data + (v1087_data * v1110_data)).copy_to(r0 + (2));
              }
              int32_t v1116_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1121_data(0.0f);
              v1121_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[44]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1123_data(0.0f);
              v1123_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v8_g);
              if (v8_g) {
                (v1123_data + (v1087_data * v1121_data)).copy_to(r0 + (3));
              }
              int32_t v1127_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1132_data(0.0f);
              v1132_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[56]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1134_data(0.0f);
              v1134_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v8_g);
              if (v8_g) {
                (v1134_data + (v1087_data * v1132_data)).copy_to(r0 + (4));
              }
              int32_t v1138_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1143_data(0.0f);
              v1143_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[68]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1145_data(0.0f);
              v1145_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v8_g);
              if (v8_g) {
                (v1145_data + (v1087_data * v1143_data)).copy_to(r0 + (5));
              }
              int32_t v1149_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1154_data(0.0f);
              v1154_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[80]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1156_data(0.0f);
              v1156_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[6]), v8_g);
              if (v8_g) {
                (v1156_data + (v1087_data * v1154_data)).copy_to(r0 + (6));
              }
              int32_t v1160_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1165_data(0.0f);
              v1165_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[92]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1167_data(0.0f);
              v1167_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[7]), v8_g);
              if (v8_g) {
                (v1167_data + (v1087_data * v1165_data)).copy_to(r0 + (7));
              }
              int32_t v1171_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1176_data(0.0f);
              v1176_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[104]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1178_data(0.0f);
              v1178_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[8]), v8_g);
              if (v8_g) {
                (v1178_data + (v1087_data * v1176_data)).copy_to(r0 + (8));
              }
              int32_t v1182_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1187_data(0.0f);
              v1187_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[116]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1189_data(0.0f);
              v1189_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[9]), v8_g);
              if (v8_g) {
                (v1189_data + (v1087_data * v1187_data)).copy_to(r0 + (9));
              }
              int32_t v1193_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1198_data(0.0f);
              v1198_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[128]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1200_data(0.0f);
              v1200_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[10]), v8_g);
              if (v8_g) {
                (v1200_data + (v1087_data * v1198_data)).copy_to(r0 + (10));
              }
              int32_t v1204_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1209_data(0.0f);
              v1209_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[140]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1211_data(0.0f);
              v1211_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[11]), v8_g);
              if (v8_g) {
                (v1211_data + (v1087_data * v1209_data)).copy_to(r0 + (11));
              }
              int32_t v1217_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v1221_data(0.0f);
              v1221_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[54_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1222_data(0.0f);
              v1222_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[9]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1224_data(0.0f);
              v1224_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v8_g);
              if (v8_g) {
                (v1224_data + (v1221_data * v1222_data)).copy_to(r0 + (0));
              }
              int32_t v1228_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v1233_data(0.0f);
              v1233_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[21]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1235_data(0.0f);
              v1235_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v8_g);
              if (v8_g) {
                (v1235_data + (v1221_data * v1233_data)).copy_to(r0 + (1));
              }
              int32_t v1239_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v1244_data(0.0f);
              v1244_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[33]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1246_data(0.0f);
              v1246_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v8_g);
              if (v8_g) {
                (v1246_data + (v1221_data * v1244_data)).copy_to(r0 + (2));
              }
              int32_t v1250_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v1255_data(0.0f);
              v1255_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[45]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1257_data(0.0f);
              v1257_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v8_g);
              if (v8_g) {
                (v1257_data + (v1221_data * v1255_data)).copy_to(r0 + (3));
              }
              int32_t v1261_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v1266_data(0.0f);
              v1266_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[57]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1268_data(0.0f);
              v1268_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v8_g);
              if (v8_g) {
                (v1268_data + (v1221_data * v1266_data)).copy_to(r0 + (4));
              }
              int32_t v1272_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v1277_data(0.0f);
              v1277_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[69]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1279_data(0.0f);
              v1279_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v8_g);
              if (v8_g) {
                (v1279_data + (v1221_data * v1277_data)).copy_to(r0 + (5));
              }
              int32_t v1283_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v1288_data(0.0f);
              v1288_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[81]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1290_data(0.0f);
              v1290_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[6]), v8_g);
              if (v8_g) {
                (v1290_data + (v1221_data * v1288_data)).copy_to(r0 + (6));
              }
              int32_t v1294_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v1299_data(0.0f);
              v1299_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[93]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1301_data(0.0f);
              v1301_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[7]), v8_g);
              if (v8_g) {
                (v1301_data + (v1221_data * v1299_data)).copy_to(r0 + (7));
              }
              int32_t v1305_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v1310_data(0.0f);
              v1310_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[105]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1312_data(0.0f);
              v1312_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[8]), v8_g);
              if (v8_g) {
                (v1312_data + (v1221_data * v1310_data)).copy_to(r0 + (8));
              }
              int32_t v1316_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v1321_data(0.0f);
              v1321_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[117]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1323_data(0.0f);
              v1323_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[9]), v8_g);
              if (v8_g) {
                (v1323_data + (v1221_data * v1321_data)).copy_to(r0 + (9));
              }
              int32_t v1327_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v1332_data(0.0f);
              v1332_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[129]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1334_data(0.0f);
              v1334_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[10]), v8_g);
              if (v8_g) {
                (v1334_data + (v1221_data * v1332_data)).copy_to(r0 + (10));
              }
              int32_t v1338_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v1343_data(0.0f);
              v1343_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[141]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1345_data(0.0f);
              v1345_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[11]), v8_g);
              if (v8_g) {
                (v1345_data + (v1221_data * v1343_data)).copy_to(r0 + (11));
              }
              int32_t v1351_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1355_data(0.0f);
              v1355_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[60_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1356_data(0.0f);
              v1356_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[10]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1358_data(0.0f);
              v1358_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v8_g);
              if (v8_g) {
                (v1358_data + (v1355_data * v1356_data)).copy_to(r0 + (0));
              }
              int32_t v1362_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1367_data(0.0f);
              v1367_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[22]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1369_data(0.0f);
              v1369_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v8_g);
              if (v8_g) {
                (v1369_data + (v1355_data * v1367_data)).copy_to(r0 + (1));
              }
              int32_t v1373_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1378_data(0.0f);
              v1378_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[34]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1380_data(0.0f);
              v1380_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v8_g);
              if (v8_g) {
                (v1380_data + (v1355_data * v1378_data)).copy_to(r0 + (2));
              }
              int32_t v1384_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1389_data(0.0f);
              v1389_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[46]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1391_data(0.0f);
              v1391_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v8_g);
              if (v8_g) {
                (v1391_data + (v1355_data * v1389_data)).copy_to(r0 + (3));
              }
              int32_t v1395_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1400_data(0.0f);
              v1400_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[58]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1402_data(0.0f);
              v1402_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v8_g);
              if (v8_g) {
                (v1402_data + (v1355_data * v1400_data)).copy_to(r0 + (4));
              }
              int32_t v1406_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1411_data(0.0f);
              v1411_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[70]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1413_data(0.0f);
              v1413_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v8_g);
              if (v8_g) {
                (v1413_data + (v1355_data * v1411_data)).copy_to(r0 + (5));
              }
              int32_t v1417_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1422_data(0.0f);
              v1422_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[82]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1424_data(0.0f);
              v1424_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[6]), v8_g);
              if (v8_g) {
                (v1424_data + (v1355_data * v1422_data)).copy_to(r0 + (6));
              }
              int32_t v1428_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1433_data(0.0f);
              v1433_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[94]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1435_data(0.0f);
              v1435_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[7]), v8_g);
              if (v8_g) {
                (v1435_data + (v1355_data * v1433_data)).copy_to(r0 + (7));
              }
              int32_t v1439_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1444_data(0.0f);
              v1444_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[106]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1446_data(0.0f);
              v1446_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[8]), v8_g);
              if (v8_g) {
                (v1446_data + (v1355_data * v1444_data)).copy_to(r0 + (8));
              }
              int32_t v1450_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1455_data(0.0f);
              v1455_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[118]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1457_data(0.0f);
              v1457_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[9]), v8_g);
              if (v8_g) {
                (v1457_data + (v1355_data * v1455_data)).copy_to(r0 + (9));
              }
              int32_t v1461_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1466_data(0.0f);
              v1466_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[130]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1468_data(0.0f);
              v1468_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[10]), v8_g);
              if (v8_g) {
                (v1468_data + (v1355_data * v1466_data)).copy_to(r0 + (10));
              }
              int32_t v1472_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1477_data(0.0f);
              v1477_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[142]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1479_data(0.0f);
              v1479_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[11]), v8_g);
              if (v8_g) {
                (v1479_data + (v1355_data * v1477_data)).copy_to(r0 + (11));
              }
              int32_t v1485_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v1489_data(0.0f);
              v1489_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[66_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1490_data(0.0f);
              v1490_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[11]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1492_data(0.0f);
              v1492_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v8_g);
              if (v8_g) {
                (v1492_data + (v1489_data * v1490_data)).copy_to(r0 + (0));
              }
              int32_t v1496_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v1501_data(0.0f);
              v1501_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[23]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1503_data(0.0f);
              v1503_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v8_g);
              if (v8_g) {
                (v1503_data + (v1489_data * v1501_data)).copy_to(r0 + (1));
              }
              int32_t v1507_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v1512_data(0.0f);
              v1512_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[35]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1514_data(0.0f);
              v1514_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v8_g);
              if (v8_g) {
                (v1514_data + (v1489_data * v1512_data)).copy_to(r0 + (2));
              }
              int32_t v1518_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v1523_data(0.0f);
              v1523_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[47]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1525_data(0.0f);
              v1525_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v8_g);
              if (v8_g) {
                (v1525_data + (v1489_data * v1523_data)).copy_to(r0 + (3));
              }
              int32_t v1529_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v1534_data(0.0f);
              v1534_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[59]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1536_data(0.0f);
              v1536_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v8_g);
              if (v8_g) {
                (v1536_data + (v1489_data * v1534_data)).copy_to(r0 + (4));
              }
              int32_t v1540_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v1545_data(0.0f);
              v1545_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[71]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1547_data(0.0f);
              v1547_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v8_g);
              if (v8_g) {
                (v1547_data + (v1489_data * v1545_data)).copy_to(r0 + (5));
              }
              int32_t v1551_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v1556_data(0.0f);
              v1556_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[83]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1558_data(0.0f);
              v1558_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[6]), v8_g);
              if (v8_g) {
                (v1558_data + (v1489_data * v1556_data)).copy_to(r0 + (6));
              }
              int32_t v1562_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v1567_data(0.0f);
              v1567_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[95]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1569_data(0.0f);
              v1569_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[7]), v8_g);
              if (v8_g) {
                (v1569_data + (v1489_data * v1567_data)).copy_to(r0 + (7));
              }
              int32_t v1573_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v1578_data(0.0f);
              v1578_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[107]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1580_data(0.0f);
              v1580_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[8]), v8_g);
              if (v8_g) {
                (v1580_data + (v1489_data * v1578_data)).copy_to(r0 + (8));
              }
              int32_t v1584_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v1589_data(0.0f);
              v1589_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[119]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1591_data(0.0f);
              v1591_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[9]), v8_g);
              if (v8_g) {
                (v1591_data + (v1489_data * v1589_data)).copy_to(r0 + (9));
              }
              int32_t v1595_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v1600_data(0.0f);
              v1600_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[131]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1602_data(0.0f);
              v1602_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[10]), v8_g);
              if (v8_g) {
                (v1602_data + (v1489_data * v1600_data)).copy_to(r0 + (10));
              }
              int32_t v1606_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v1611_data(0.0f);
              v1611_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[143]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1613_data(0.0f);
              v1613_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[11]), v8_g);
              if (v8_g) {
                (v1613_data + (v1489_data * v1611_data)).copy_to(r0 + (11));
              }
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r0);
              #pragma unroll
              for (int32_t v1618_i1 = 0; v1618_i1 < 12; ++v1618_i1) {
                int32_t v1619_a = 0 + v1618_i1;
                tensorforge::intel_esimd::simd<float, 16> v1621_data(0.0f);
                v1621_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[v1618_i1]), v8_g);
                int32_t v1625_a = 0_i32 + (v1618_i1 * 12);
                if (v8_g) {
                  s1[v1625_a] = v1621_data;
                }
              }
              float r1[12]{};
              // r1 = +(glb_m2 * s0) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              float ir1[12]{};
              int32_t v1632_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1636_data(0.0f);
              v1636_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m2[0_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1637_data(0.0f);
              v1637_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[0]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1639_data(0.0f);
              v1639_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v8_g);
              if (v8_g) {
                (v1639_data + (v1636_data * v1637_data)).copy_to(ir1 + (0));
              }
              int32_t v1643_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1648_data(0.0f);
              v1648_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[12]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1650_data(0.0f);
              v1650_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v8_g);
              if (v8_g) {
                (v1650_data + (v1636_data * v1648_data)).copy_to(ir1 + (1));
              }
              int32_t v1654_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1659_data(0.0f);
              v1659_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[24]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1661_data(0.0f);
              v1661_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v8_g);
              if (v8_g) {
                (v1661_data + (v1636_data * v1659_data)).copy_to(ir1 + (2));
              }
              int32_t v1665_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1670_data(0.0f);
              v1670_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[36]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1672_data(0.0f);
              v1672_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v8_g);
              if (v8_g) {
                (v1672_data + (v1636_data * v1670_data)).copy_to(ir1 + (3));
              }
              int32_t v1676_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1681_data(0.0f);
              v1681_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[48]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1683_data(0.0f);
              v1683_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v8_g);
              if (v8_g) {
                (v1683_data + (v1636_data * v1681_data)).copy_to(ir1 + (4));
              }
              int32_t v1687_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1692_data(0.0f);
              v1692_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[60]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1694_data(0.0f);
              v1694_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v8_g);
              if (v8_g) {
                (v1694_data + (v1636_data * v1692_data)).copy_to(ir1 + (5));
              }
              int32_t v1698_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1703_data(0.0f);
              v1703_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[72]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1705_data(0.0f);
              v1705_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v8_g);
              if (v8_g) {
                (v1705_data + (v1636_data * v1703_data)).copy_to(ir1 + (6));
              }
              int32_t v1709_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1714_data(0.0f);
              v1714_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[84]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1716_data(0.0f);
              v1716_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v8_g);
              if (v8_g) {
                (v1716_data + (v1636_data * v1714_data)).copy_to(ir1 + (7));
              }
              int32_t v1720_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1725_data(0.0f);
              v1725_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[96]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1727_data(0.0f);
              v1727_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[8]), v8_g);
              if (v8_g) {
                (v1727_data + (v1636_data * v1725_data)).copy_to(ir1 + (8));
              }
              int32_t v1731_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1736_data(0.0f);
              v1736_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[108]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1738_data(0.0f);
              v1738_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[9]), v8_g);
              if (v8_g) {
                (v1738_data + (v1636_data * v1736_data)).copy_to(ir1 + (9));
              }
              int32_t v1742_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1747_data(0.0f);
              v1747_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[120]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1749_data(0.0f);
              v1749_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[10]), v8_g);
              if (v8_g) {
                (v1749_data + (v1636_data * v1747_data)).copy_to(ir1 + (10));
              }
              int32_t v1753_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1758_data(0.0f);
              v1758_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[132]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1760_data(0.0f);
              v1760_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[11]), v8_g);
              if (v8_g) {
                (v1760_data + (v1636_data * v1758_data)).copy_to(ir1 + (11));
              }
              int32_t v1766_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v1770_data(0.0f);
              v1770_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m2[6_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1771_data(0.0f);
              v1771_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[1]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1773_data(0.0f);
              v1773_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v8_g);
              if (v8_g) {
                (v1773_data + (v1770_data * v1771_data)).copy_to(ir1 + (0));
              }
              int32_t v1777_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v1782_data(0.0f);
              v1782_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[13]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1784_data(0.0f);
              v1784_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v8_g);
              if (v8_g) {
                (v1784_data + (v1770_data * v1782_data)).copy_to(ir1 + (1));
              }
              int32_t v1788_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v1793_data(0.0f);
              v1793_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[25]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1795_data(0.0f);
              v1795_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v8_g);
              if (v8_g) {
                (v1795_data + (v1770_data * v1793_data)).copy_to(ir1 + (2));
              }
              int32_t v1799_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v1804_data(0.0f);
              v1804_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[37]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1806_data(0.0f);
              v1806_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v8_g);
              if (v8_g) {
                (v1806_data + (v1770_data * v1804_data)).copy_to(ir1 + (3));
              }
              int32_t v1810_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v1815_data(0.0f);
              v1815_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[49]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1817_data(0.0f);
              v1817_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v8_g);
              if (v8_g) {
                (v1817_data + (v1770_data * v1815_data)).copy_to(ir1 + (4));
              }
              int32_t v1821_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v1826_data(0.0f);
              v1826_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[61]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1828_data(0.0f);
              v1828_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v8_g);
              if (v8_g) {
                (v1828_data + (v1770_data * v1826_data)).copy_to(ir1 + (5));
              }
              int32_t v1832_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v1837_data(0.0f);
              v1837_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[73]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1839_data(0.0f);
              v1839_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v8_g);
              if (v8_g) {
                (v1839_data + (v1770_data * v1837_data)).copy_to(ir1 + (6));
              }
              int32_t v1843_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v1848_data(0.0f);
              v1848_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[85]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1850_data(0.0f);
              v1850_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v8_g);
              if (v8_g) {
                (v1850_data + (v1770_data * v1848_data)).copy_to(ir1 + (7));
              }
              int32_t v1854_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v1859_data(0.0f);
              v1859_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[97]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1861_data(0.0f);
              v1861_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[8]), v8_g);
              if (v8_g) {
                (v1861_data + (v1770_data * v1859_data)).copy_to(ir1 + (8));
              }
              int32_t v1865_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v1870_data(0.0f);
              v1870_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[109]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1872_data(0.0f);
              v1872_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[9]), v8_g);
              if (v8_g) {
                (v1872_data + (v1770_data * v1870_data)).copy_to(ir1 + (9));
              }
              int32_t v1876_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v1881_data(0.0f);
              v1881_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[121]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1883_data(0.0f);
              v1883_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[10]), v8_g);
              if (v8_g) {
                (v1883_data + (v1770_data * v1881_data)).copy_to(ir1 + (10));
              }
              int32_t v1887_a = 0_i32 + 6;
              tensorforge::intel_esimd::simd<float, 16> v1892_data(0.0f);
              v1892_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[133]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1894_data(0.0f);
              v1894_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[11]), v8_g);
              if (v8_g) {
                (v1894_data + (v1770_data * v1892_data)).copy_to(ir1 + (11));
              }
              int32_t v1900_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1904_data(0.0f);
              v1904_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m2[12_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1905_data(0.0f);
              v1905_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[2]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1907_data(0.0f);
              v1907_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v8_g);
              if (v8_g) {
                (v1907_data + (v1904_data * v1905_data)).copy_to(ir1 + (0));
              }
              int32_t v1911_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1916_data(0.0f);
              v1916_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[14]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1918_data(0.0f);
              v1918_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v8_g);
              if (v8_g) {
                (v1918_data + (v1904_data * v1916_data)).copy_to(ir1 + (1));
              }
              int32_t v1922_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1927_data(0.0f);
              v1927_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[26]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1929_data(0.0f);
              v1929_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v8_g);
              if (v8_g) {
                (v1929_data + (v1904_data * v1927_data)).copy_to(ir1 + (2));
              }
              int32_t v1933_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1938_data(0.0f);
              v1938_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[38]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1940_data(0.0f);
              v1940_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v8_g);
              if (v8_g) {
                (v1940_data + (v1904_data * v1938_data)).copy_to(ir1 + (3));
              }
              int32_t v1944_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1949_data(0.0f);
              v1949_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[50]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1951_data(0.0f);
              v1951_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v8_g);
              if (v8_g) {
                (v1951_data + (v1904_data * v1949_data)).copy_to(ir1 + (4));
              }
              int32_t v1955_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1960_data(0.0f);
              v1960_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[62]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1962_data(0.0f);
              v1962_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v8_g);
              if (v8_g) {
                (v1962_data + (v1904_data * v1960_data)).copy_to(ir1 + (5));
              }
              int32_t v1966_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1971_data(0.0f);
              v1971_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[74]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1973_data(0.0f);
              v1973_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v8_g);
              if (v8_g) {
                (v1973_data + (v1904_data * v1971_data)).copy_to(ir1 + (6));
              }
              int32_t v1977_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1982_data(0.0f);
              v1982_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[86]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1984_data(0.0f);
              v1984_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v8_g);
              if (v8_g) {
                (v1984_data + (v1904_data * v1982_data)).copy_to(ir1 + (7));
              }
              int32_t v1988_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1993_data(0.0f);
              v1993_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[98]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v1995_data(0.0f);
              v1995_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[8]), v8_g);
              if (v8_g) {
                (v1995_data + (v1904_data * v1993_data)).copy_to(ir1 + (8));
              }
              int32_t v1999_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v2004_data(0.0f);
              v2004_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[110]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2006_data(0.0f);
              v2006_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[9]), v8_g);
              if (v8_g) {
                (v2006_data + (v1904_data * v2004_data)).copy_to(ir1 + (9));
              }
              int32_t v2010_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v2015_data(0.0f);
              v2015_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[122]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2017_data(0.0f);
              v2017_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[10]), v8_g);
              if (v8_g) {
                (v2017_data + (v1904_data * v2015_data)).copy_to(ir1 + (10));
              }
              int32_t v2021_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v2026_data(0.0f);
              v2026_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[134]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2028_data(0.0f);
              v2028_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[11]), v8_g);
              if (v8_g) {
                (v2028_data + (v1904_data * v2026_data)).copy_to(ir1 + (11));
              }
              int32_t v2034_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v2038_data(0.0f);
              v2038_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m2[18_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2039_data(0.0f);
              v2039_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[3]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2041_data(0.0f);
              v2041_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v8_g);
              if (v8_g) {
                (v2041_data + (v2038_data * v2039_data)).copy_to(ir1 + (0));
              }
              int32_t v2045_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v2050_data(0.0f);
              v2050_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[15]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2052_data(0.0f);
              v2052_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v8_g);
              if (v8_g) {
                (v2052_data + (v2038_data * v2050_data)).copy_to(ir1 + (1));
              }
              int32_t v2056_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v2061_data(0.0f);
              v2061_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[27]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2063_data(0.0f);
              v2063_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v8_g);
              if (v8_g) {
                (v2063_data + (v2038_data * v2061_data)).copy_to(ir1 + (2));
              }
              int32_t v2067_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v2072_data(0.0f);
              v2072_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[39]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2074_data(0.0f);
              v2074_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v8_g);
              if (v8_g) {
                (v2074_data + (v2038_data * v2072_data)).copy_to(ir1 + (3));
              }
              int32_t v2078_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v2083_data(0.0f);
              v2083_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[51]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2085_data(0.0f);
              v2085_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v8_g);
              if (v8_g) {
                (v2085_data + (v2038_data * v2083_data)).copy_to(ir1 + (4));
              }
              int32_t v2089_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v2094_data(0.0f);
              v2094_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[63]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2096_data(0.0f);
              v2096_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v8_g);
              if (v8_g) {
                (v2096_data + (v2038_data * v2094_data)).copy_to(ir1 + (5));
              }
              int32_t v2100_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v2105_data(0.0f);
              v2105_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[75]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2107_data(0.0f);
              v2107_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v8_g);
              if (v8_g) {
                (v2107_data + (v2038_data * v2105_data)).copy_to(ir1 + (6));
              }
              int32_t v2111_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v2116_data(0.0f);
              v2116_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[87]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2118_data(0.0f);
              v2118_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v8_g);
              if (v8_g) {
                (v2118_data + (v2038_data * v2116_data)).copy_to(ir1 + (7));
              }
              int32_t v2122_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v2127_data(0.0f);
              v2127_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[99]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2129_data(0.0f);
              v2129_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[8]), v8_g);
              if (v8_g) {
                (v2129_data + (v2038_data * v2127_data)).copy_to(ir1 + (8));
              }
              int32_t v2133_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v2138_data(0.0f);
              v2138_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[111]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2140_data(0.0f);
              v2140_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[9]), v8_g);
              if (v8_g) {
                (v2140_data + (v2038_data * v2138_data)).copy_to(ir1 + (9));
              }
              int32_t v2144_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v2149_data(0.0f);
              v2149_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[123]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2151_data(0.0f);
              v2151_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[10]), v8_g);
              if (v8_g) {
                (v2151_data + (v2038_data * v2149_data)).copy_to(ir1 + (10));
              }
              int32_t v2155_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v2160_data(0.0f);
              v2160_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[135]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2162_data(0.0f);
              v2162_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[11]), v8_g);
              if (v8_g) {
                (v2162_data + (v2038_data * v2160_data)).copy_to(ir1 + (11));
              }
              int32_t v2168_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2172_data(0.0f);
              v2172_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m2[24_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2173_data(0.0f);
              v2173_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[4]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2175_data(0.0f);
              v2175_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v8_g);
              if (v8_g) {
                (v2175_data + (v2172_data * v2173_data)).copy_to(ir1 + (0));
              }
              int32_t v2179_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2184_data(0.0f);
              v2184_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[16]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2186_data(0.0f);
              v2186_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v8_g);
              if (v8_g) {
                (v2186_data + (v2172_data * v2184_data)).copy_to(ir1 + (1));
              }
              int32_t v2190_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2195_data(0.0f);
              v2195_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[28]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2197_data(0.0f);
              v2197_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v8_g);
              if (v8_g) {
                (v2197_data + (v2172_data * v2195_data)).copy_to(ir1 + (2));
              }
              int32_t v2201_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2206_data(0.0f);
              v2206_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[40]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2208_data(0.0f);
              v2208_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v8_g);
              if (v8_g) {
                (v2208_data + (v2172_data * v2206_data)).copy_to(ir1 + (3));
              }
              int32_t v2212_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2217_data(0.0f);
              v2217_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[52]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2219_data(0.0f);
              v2219_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v8_g);
              if (v8_g) {
                (v2219_data + (v2172_data * v2217_data)).copy_to(ir1 + (4));
              }
              int32_t v2223_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2228_data(0.0f);
              v2228_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[64]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2230_data(0.0f);
              v2230_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v8_g);
              if (v8_g) {
                (v2230_data + (v2172_data * v2228_data)).copy_to(ir1 + (5));
              }
              int32_t v2234_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2239_data(0.0f);
              v2239_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[76]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2241_data(0.0f);
              v2241_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v8_g);
              if (v8_g) {
                (v2241_data + (v2172_data * v2239_data)).copy_to(ir1 + (6));
              }
              int32_t v2245_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2250_data(0.0f);
              v2250_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[88]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2252_data(0.0f);
              v2252_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v8_g);
              if (v8_g) {
                (v2252_data + (v2172_data * v2250_data)).copy_to(ir1 + (7));
              }
              int32_t v2256_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2261_data(0.0f);
              v2261_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[100]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2263_data(0.0f);
              v2263_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[8]), v8_g);
              if (v8_g) {
                (v2263_data + (v2172_data * v2261_data)).copy_to(ir1 + (8));
              }
              int32_t v2267_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2272_data(0.0f);
              v2272_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[112]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2274_data(0.0f);
              v2274_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[9]), v8_g);
              if (v8_g) {
                (v2274_data + (v2172_data * v2272_data)).copy_to(ir1 + (9));
              }
              int32_t v2278_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2283_data(0.0f);
              v2283_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[124]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2285_data(0.0f);
              v2285_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[10]), v8_g);
              if (v8_g) {
                (v2285_data + (v2172_data * v2283_data)).copy_to(ir1 + (10));
              }
              int32_t v2289_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2294_data(0.0f);
              v2294_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[136]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2296_data(0.0f);
              v2296_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[11]), v8_g);
              if (v8_g) {
                (v2296_data + (v2172_data * v2294_data)).copy_to(ir1 + (11));
              }
              int32_t v2302_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v2306_data(0.0f);
              v2306_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m2[30_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2307_data(0.0f);
              v2307_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[5]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2309_data(0.0f);
              v2309_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v8_g);
              if (v8_g) {
                (v2309_data + (v2306_data * v2307_data)).copy_to(ir1 + (0));
              }
              int32_t v2313_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v2318_data(0.0f);
              v2318_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[17]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2320_data(0.0f);
              v2320_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v8_g);
              if (v8_g) {
                (v2320_data + (v2306_data * v2318_data)).copy_to(ir1 + (1));
              }
              int32_t v2324_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v2329_data(0.0f);
              v2329_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[29]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2331_data(0.0f);
              v2331_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v8_g);
              if (v8_g) {
                (v2331_data + (v2306_data * v2329_data)).copy_to(ir1 + (2));
              }
              int32_t v2335_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v2340_data(0.0f);
              v2340_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[41]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2342_data(0.0f);
              v2342_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v8_g);
              if (v8_g) {
                (v2342_data + (v2306_data * v2340_data)).copy_to(ir1 + (3));
              }
              int32_t v2346_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v2351_data(0.0f);
              v2351_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[53]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2353_data(0.0f);
              v2353_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v8_g);
              if (v8_g) {
                (v2353_data + (v2306_data * v2351_data)).copy_to(ir1 + (4));
              }
              int32_t v2357_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v2362_data(0.0f);
              v2362_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[65]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2364_data(0.0f);
              v2364_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v8_g);
              if (v8_g) {
                (v2364_data + (v2306_data * v2362_data)).copy_to(ir1 + (5));
              }
              int32_t v2368_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v2373_data(0.0f);
              v2373_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[77]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2375_data(0.0f);
              v2375_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v8_g);
              if (v8_g) {
                (v2375_data + (v2306_data * v2373_data)).copy_to(ir1 + (6));
              }
              int32_t v2379_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v2384_data(0.0f);
              v2384_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[89]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2386_data(0.0f);
              v2386_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v8_g);
              if (v8_g) {
                (v2386_data + (v2306_data * v2384_data)).copy_to(ir1 + (7));
              }
              int32_t v2390_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v2395_data(0.0f);
              v2395_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[101]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2397_data(0.0f);
              v2397_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[8]), v8_g);
              if (v8_g) {
                (v2397_data + (v2306_data * v2395_data)).copy_to(ir1 + (8));
              }
              int32_t v2401_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v2406_data(0.0f);
              v2406_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[113]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2408_data(0.0f);
              v2408_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[9]), v8_g);
              if (v8_g) {
                (v2408_data + (v2306_data * v2406_data)).copy_to(ir1 + (9));
              }
              int32_t v2412_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v2417_data(0.0f);
              v2417_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[125]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2419_data(0.0f);
              v2419_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[10]), v8_g);
              if (v8_g) {
                (v2419_data + (v2306_data * v2417_data)).copy_to(ir1 + (10));
              }
              int32_t v2423_a = 0_i32 + 30;
              tensorforge::intel_esimd::simd<float, 16> v2428_data(0.0f);
              v2428_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[137]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2430_data(0.0f);
              v2430_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[11]), v8_g);
              if (v8_g) {
                (v2430_data + (v2306_data * v2428_data)).copy_to(ir1 + (11));
              }
              int32_t v2436_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2440_data(0.0f);
              v2440_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m2[36_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2441_data(0.0f);
              v2441_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[6]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2443_data(0.0f);
              v2443_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v8_g);
              if (v8_g) {
                (v2443_data + (v2440_data * v2441_data)).copy_to(ir1 + (0));
              }
              int32_t v2447_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2452_data(0.0f);
              v2452_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[18]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2454_data(0.0f);
              v2454_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v8_g);
              if (v8_g) {
                (v2454_data + (v2440_data * v2452_data)).copy_to(ir1 + (1));
              }
              int32_t v2458_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2463_data(0.0f);
              v2463_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[30]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2465_data(0.0f);
              v2465_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v8_g);
              if (v8_g) {
                (v2465_data + (v2440_data * v2463_data)).copy_to(ir1 + (2));
              }
              int32_t v2469_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2474_data(0.0f);
              v2474_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[42]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2476_data(0.0f);
              v2476_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v8_g);
              if (v8_g) {
                (v2476_data + (v2440_data * v2474_data)).copy_to(ir1 + (3));
              }
              int32_t v2480_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2485_data(0.0f);
              v2485_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[54]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2487_data(0.0f);
              v2487_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v8_g);
              if (v8_g) {
                (v2487_data + (v2440_data * v2485_data)).copy_to(ir1 + (4));
              }
              int32_t v2491_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2496_data(0.0f);
              v2496_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[66]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2498_data(0.0f);
              v2498_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v8_g);
              if (v8_g) {
                (v2498_data + (v2440_data * v2496_data)).copy_to(ir1 + (5));
              }
              int32_t v2502_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2507_data(0.0f);
              v2507_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[78]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2509_data(0.0f);
              v2509_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v8_g);
              if (v8_g) {
                (v2509_data + (v2440_data * v2507_data)).copy_to(ir1 + (6));
              }
              int32_t v2513_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2518_data(0.0f);
              v2518_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[90]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2520_data(0.0f);
              v2520_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v8_g);
              if (v8_g) {
                (v2520_data + (v2440_data * v2518_data)).copy_to(ir1 + (7));
              }
              int32_t v2524_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2529_data(0.0f);
              v2529_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[102]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2531_data(0.0f);
              v2531_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[8]), v8_g);
              if (v8_g) {
                (v2531_data + (v2440_data * v2529_data)).copy_to(ir1 + (8));
              }
              int32_t v2535_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2540_data(0.0f);
              v2540_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[114]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2542_data(0.0f);
              v2542_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[9]), v8_g);
              if (v8_g) {
                (v2542_data + (v2440_data * v2540_data)).copy_to(ir1 + (9));
              }
              int32_t v2546_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2551_data(0.0f);
              v2551_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[126]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2553_data(0.0f);
              v2553_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[10]), v8_g);
              if (v8_g) {
                (v2553_data + (v2440_data * v2551_data)).copy_to(ir1 + (10));
              }
              int32_t v2557_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2562_data(0.0f);
              v2562_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[138]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2564_data(0.0f);
              v2564_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[11]), v8_g);
              if (v8_g) {
                (v2564_data + (v2440_data * v2562_data)).copy_to(ir1 + (11));
              }
              int32_t v2570_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v2574_data(0.0f);
              v2574_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m2[42_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2575_data(0.0f);
              v2575_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[7]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2577_data(0.0f);
              v2577_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v8_g);
              if (v8_g) {
                (v2577_data + (v2574_data * v2575_data)).copy_to(ir1 + (0));
              }
              int32_t v2581_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v2586_data(0.0f);
              v2586_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[19]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2588_data(0.0f);
              v2588_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v8_g);
              if (v8_g) {
                (v2588_data + (v2574_data * v2586_data)).copy_to(ir1 + (1));
              }
              int32_t v2592_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v2597_data(0.0f);
              v2597_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[31]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2599_data(0.0f);
              v2599_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v8_g);
              if (v8_g) {
                (v2599_data + (v2574_data * v2597_data)).copy_to(ir1 + (2));
              }
              int32_t v2603_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v2608_data(0.0f);
              v2608_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[43]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2610_data(0.0f);
              v2610_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v8_g);
              if (v8_g) {
                (v2610_data + (v2574_data * v2608_data)).copy_to(ir1 + (3));
              }
              int32_t v2614_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v2619_data(0.0f);
              v2619_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[55]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2621_data(0.0f);
              v2621_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v8_g);
              if (v8_g) {
                (v2621_data + (v2574_data * v2619_data)).copy_to(ir1 + (4));
              }
              int32_t v2625_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v2630_data(0.0f);
              v2630_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[67]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2632_data(0.0f);
              v2632_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v8_g);
              if (v8_g) {
                (v2632_data + (v2574_data * v2630_data)).copy_to(ir1 + (5));
              }
              int32_t v2636_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v2641_data(0.0f);
              v2641_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[79]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2643_data(0.0f);
              v2643_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v8_g);
              if (v8_g) {
                (v2643_data + (v2574_data * v2641_data)).copy_to(ir1 + (6));
              }
              int32_t v2647_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v2652_data(0.0f);
              v2652_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[91]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2654_data(0.0f);
              v2654_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v8_g);
              if (v8_g) {
                (v2654_data + (v2574_data * v2652_data)).copy_to(ir1 + (7));
              }
              int32_t v2658_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v2663_data(0.0f);
              v2663_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[103]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2665_data(0.0f);
              v2665_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[8]), v8_g);
              if (v8_g) {
                (v2665_data + (v2574_data * v2663_data)).copy_to(ir1 + (8));
              }
              int32_t v2669_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v2674_data(0.0f);
              v2674_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[115]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2676_data(0.0f);
              v2676_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[9]), v8_g);
              if (v8_g) {
                (v2676_data + (v2574_data * v2674_data)).copy_to(ir1 + (9));
              }
              int32_t v2680_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v2685_data(0.0f);
              v2685_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[127]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2687_data(0.0f);
              v2687_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[10]), v8_g);
              if (v8_g) {
                (v2687_data + (v2574_data * v2685_data)).copy_to(ir1 + (10));
              }
              int32_t v2691_a = 0_i32 + 42;
              tensorforge::intel_esimd::simd<float, 16> v2696_data(0.0f);
              v2696_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[139]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2698_data(0.0f);
              v2698_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[11]), v8_g);
              if (v8_g) {
                (v2698_data + (v2574_data * v2696_data)).copy_to(ir1 + (11));
              }
              int32_t v2704_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2708_data(0.0f);
              v2708_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m2[48_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2709_data(0.0f);
              v2709_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[8]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2711_data(0.0f);
              v2711_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v8_g);
              if (v8_g) {
                (v2711_data + (v2708_data * v2709_data)).copy_to(ir1 + (0));
              }
              int32_t v2715_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2720_data(0.0f);
              v2720_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[20]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2722_data(0.0f);
              v2722_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v8_g);
              if (v8_g) {
                (v2722_data + (v2708_data * v2720_data)).copy_to(ir1 + (1));
              }
              int32_t v2726_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2731_data(0.0f);
              v2731_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2733_data(0.0f);
              v2733_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v8_g);
              if (v8_g) {
                (v2733_data + (v2708_data * v2731_data)).copy_to(ir1 + (2));
              }
              int32_t v2737_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2742_data(0.0f);
              v2742_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[44]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2744_data(0.0f);
              v2744_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v8_g);
              if (v8_g) {
                (v2744_data + (v2708_data * v2742_data)).copy_to(ir1 + (3));
              }
              int32_t v2748_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2753_data(0.0f);
              v2753_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[56]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2755_data(0.0f);
              v2755_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v8_g);
              if (v8_g) {
                (v2755_data + (v2708_data * v2753_data)).copy_to(ir1 + (4));
              }
              int32_t v2759_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2764_data(0.0f);
              v2764_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[68]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2766_data(0.0f);
              v2766_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v8_g);
              if (v8_g) {
                (v2766_data + (v2708_data * v2764_data)).copy_to(ir1 + (5));
              }
              int32_t v2770_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2775_data(0.0f);
              v2775_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[80]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2777_data(0.0f);
              v2777_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v8_g);
              if (v8_g) {
                (v2777_data + (v2708_data * v2775_data)).copy_to(ir1 + (6));
              }
              int32_t v2781_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2786_data(0.0f);
              v2786_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[92]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2788_data(0.0f);
              v2788_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v8_g);
              if (v8_g) {
                (v2788_data + (v2708_data * v2786_data)).copy_to(ir1 + (7));
              }
              int32_t v2792_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2797_data(0.0f);
              v2797_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[104]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2799_data(0.0f);
              v2799_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[8]), v8_g);
              if (v8_g) {
                (v2799_data + (v2708_data * v2797_data)).copy_to(ir1 + (8));
              }
              int32_t v2803_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2808_data(0.0f);
              v2808_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[116]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2810_data(0.0f);
              v2810_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[9]), v8_g);
              if (v8_g) {
                (v2810_data + (v2708_data * v2808_data)).copy_to(ir1 + (9));
              }
              int32_t v2814_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2819_data(0.0f);
              v2819_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[128]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2821_data(0.0f);
              v2821_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[10]), v8_g);
              if (v8_g) {
                (v2821_data + (v2708_data * v2819_data)).copy_to(ir1 + (10));
              }
              int32_t v2825_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2830_data(0.0f);
              v2830_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[140]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2832_data(0.0f);
              v2832_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[11]), v8_g);
              if (v8_g) {
                (v2832_data + (v2708_data * v2830_data)).copy_to(ir1 + (11));
              }
              int32_t v2838_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v2842_data(0.0f);
              v2842_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m2[54_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2843_data(0.0f);
              v2843_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[9]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2845_data(0.0f);
              v2845_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v8_g);
              if (v8_g) {
                (v2845_data + (v2842_data * v2843_data)).copy_to(ir1 + (0));
              }
              int32_t v2849_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v2854_data(0.0f);
              v2854_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[21]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2856_data(0.0f);
              v2856_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v8_g);
              if (v8_g) {
                (v2856_data + (v2842_data * v2854_data)).copy_to(ir1 + (1));
              }
              int32_t v2860_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v2865_data(0.0f);
              v2865_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[33]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2867_data(0.0f);
              v2867_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v8_g);
              if (v8_g) {
                (v2867_data + (v2842_data * v2865_data)).copy_to(ir1 + (2));
              }
              int32_t v2871_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v2876_data(0.0f);
              v2876_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[45]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2878_data(0.0f);
              v2878_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v8_g);
              if (v8_g) {
                (v2878_data + (v2842_data * v2876_data)).copy_to(ir1 + (3));
              }
              int32_t v2882_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v2887_data(0.0f);
              v2887_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[57]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2889_data(0.0f);
              v2889_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v8_g);
              if (v8_g) {
                (v2889_data + (v2842_data * v2887_data)).copy_to(ir1 + (4));
              }
              int32_t v2893_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v2898_data(0.0f);
              v2898_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[69]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2900_data(0.0f);
              v2900_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v8_g);
              if (v8_g) {
                (v2900_data + (v2842_data * v2898_data)).copy_to(ir1 + (5));
              }
              int32_t v2904_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v2909_data(0.0f);
              v2909_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[81]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2911_data(0.0f);
              v2911_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v8_g);
              if (v8_g) {
                (v2911_data + (v2842_data * v2909_data)).copy_to(ir1 + (6));
              }
              int32_t v2915_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v2920_data(0.0f);
              v2920_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[93]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2922_data(0.0f);
              v2922_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v8_g);
              if (v8_g) {
                (v2922_data + (v2842_data * v2920_data)).copy_to(ir1 + (7));
              }
              int32_t v2926_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v2931_data(0.0f);
              v2931_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[105]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2933_data(0.0f);
              v2933_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[8]), v8_g);
              if (v8_g) {
                (v2933_data + (v2842_data * v2931_data)).copy_to(ir1 + (8));
              }
              int32_t v2937_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v2942_data(0.0f);
              v2942_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[117]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2944_data(0.0f);
              v2944_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[9]), v8_g);
              if (v8_g) {
                (v2944_data + (v2842_data * v2942_data)).copy_to(ir1 + (9));
              }
              int32_t v2948_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v2953_data(0.0f);
              v2953_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[129]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2955_data(0.0f);
              v2955_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[10]), v8_g);
              if (v8_g) {
                (v2955_data + (v2842_data * v2953_data)).copy_to(ir1 + (10));
              }
              int32_t v2959_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v2964_data(0.0f);
              v2964_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[141]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2966_data(0.0f);
              v2966_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[11]), v8_g);
              if (v8_g) {
                (v2966_data + (v2842_data * v2964_data)).copy_to(ir1 + (11));
              }
              int32_t v2972_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v2976_data(0.0f);
              v2976_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m2[60_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2977_data(0.0f);
              v2977_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[10]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2979_data(0.0f);
              v2979_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v8_g);
              if (v8_g) {
                (v2979_data + (v2976_data * v2977_data)).copy_to(ir1 + (0));
              }
              int32_t v2983_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v2988_data(0.0f);
              v2988_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[22]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v2990_data(0.0f);
              v2990_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v8_g);
              if (v8_g) {
                (v2990_data + (v2976_data * v2988_data)).copy_to(ir1 + (1));
              }
              int32_t v2994_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v2999_data(0.0f);
              v2999_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[34]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3001_data(0.0f);
              v3001_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v8_g);
              if (v8_g) {
                (v3001_data + (v2976_data * v2999_data)).copy_to(ir1 + (2));
              }
              int32_t v3005_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3010_data(0.0f);
              v3010_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[46]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3012_data(0.0f);
              v3012_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v8_g);
              if (v8_g) {
                (v3012_data + (v2976_data * v3010_data)).copy_to(ir1 + (3));
              }
              int32_t v3016_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3021_data(0.0f);
              v3021_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[58]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3023_data(0.0f);
              v3023_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v8_g);
              if (v8_g) {
                (v3023_data + (v2976_data * v3021_data)).copy_to(ir1 + (4));
              }
              int32_t v3027_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3032_data(0.0f);
              v3032_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[70]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3034_data(0.0f);
              v3034_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v8_g);
              if (v8_g) {
                (v3034_data + (v2976_data * v3032_data)).copy_to(ir1 + (5));
              }
              int32_t v3038_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3043_data(0.0f);
              v3043_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[82]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3045_data(0.0f);
              v3045_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v8_g);
              if (v8_g) {
                (v3045_data + (v2976_data * v3043_data)).copy_to(ir1 + (6));
              }
              int32_t v3049_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3054_data(0.0f);
              v3054_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[94]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3056_data(0.0f);
              v3056_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v8_g);
              if (v8_g) {
                (v3056_data + (v2976_data * v3054_data)).copy_to(ir1 + (7));
              }
              int32_t v3060_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3065_data(0.0f);
              v3065_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[106]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3067_data(0.0f);
              v3067_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[8]), v8_g);
              if (v8_g) {
                (v3067_data + (v2976_data * v3065_data)).copy_to(ir1 + (8));
              }
              int32_t v3071_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3076_data(0.0f);
              v3076_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[118]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3078_data(0.0f);
              v3078_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[9]), v8_g);
              if (v8_g) {
                (v3078_data + (v2976_data * v3076_data)).copy_to(ir1 + (9));
              }
              int32_t v3082_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3087_data(0.0f);
              v3087_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[130]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3089_data(0.0f);
              v3089_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[10]), v8_g);
              if (v8_g) {
                (v3089_data + (v2976_data * v3087_data)).copy_to(ir1 + (10));
              }
              int32_t v3093_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3098_data(0.0f);
              v3098_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[142]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3100_data(0.0f);
              v3100_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[11]), v8_g);
              if (v8_g) {
                (v3100_data + (v2976_data * v3098_data)).copy_to(ir1 + (11));
              }
              int32_t v3106_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v3110_data(0.0f);
              v3110_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m2[66_i32]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3111_data(0.0f);
              v3111_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[11]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3113_data(0.0f);
              v3113_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v8_g);
              if (v8_g) {
                (v3113_data + (v3110_data * v3111_data)).copy_to(ir1 + (0));
              }
              int32_t v3117_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v3122_data(0.0f);
              v3122_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[23]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3124_data(0.0f);
              v3124_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v8_g);
              if (v8_g) {
                (v3124_data + (v3110_data * v3122_data)).copy_to(ir1 + (1));
              }
              int32_t v3128_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v3133_data(0.0f);
              v3133_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[35]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3135_data(0.0f);
              v3135_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v8_g);
              if (v8_g) {
                (v3135_data + (v3110_data * v3133_data)).copy_to(ir1 + (2));
              }
              int32_t v3139_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v3144_data(0.0f);
              v3144_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[47]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3146_data(0.0f);
              v3146_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v8_g);
              if (v8_g) {
                (v3146_data + (v3110_data * v3144_data)).copy_to(ir1 + (3));
              }
              int32_t v3150_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v3155_data(0.0f);
              v3155_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[59]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3157_data(0.0f);
              v3157_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v8_g);
              if (v8_g) {
                (v3157_data + (v3110_data * v3155_data)).copy_to(ir1 + (4));
              }
              int32_t v3161_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v3166_data(0.0f);
              v3166_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[71]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3168_data(0.0f);
              v3168_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v8_g);
              if (v8_g) {
                (v3168_data + (v3110_data * v3166_data)).copy_to(ir1 + (5));
              }
              int32_t v3172_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v3177_data(0.0f);
              v3177_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[83]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3179_data(0.0f);
              v3179_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v8_g);
              if (v8_g) {
                (v3179_data + (v3110_data * v3177_data)).copy_to(ir1 + (6));
              }
              int32_t v3183_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v3188_data(0.0f);
              v3188_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[95]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3190_data(0.0f);
              v3190_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v8_g);
              if (v8_g) {
                (v3190_data + (v3110_data * v3188_data)).copy_to(ir1 + (7));
              }
              int32_t v3194_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v3199_data(0.0f);
              v3199_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[107]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3201_data(0.0f);
              v3201_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[8]), v8_g);
              if (v8_g) {
                (v3201_data + (v3110_data * v3199_data)).copy_to(ir1 + (8));
              }
              int32_t v3205_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v3210_data(0.0f);
              v3210_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[119]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3212_data(0.0f);
              v3212_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[9]), v8_g);
              if (v8_g) {
                (v3212_data + (v3110_data * v3210_data)).copy_to(ir1 + (9));
              }
              int32_t v3216_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v3221_data(0.0f);
              v3221_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[131]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3223_data(0.0f);
              v3223_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[10]), v8_g);
              if (v8_g) {
                (v3223_data + (v3110_data * v3221_data)).copy_to(ir1 + (10));
              }
              int32_t v3227_a = 0_i32 + 66;
              tensorforge::intel_esimd::simd<float, 16> v3232_data(0.0f);
              v3232_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[143]), v8_g);
              tensorforge::intel_esimd::simd<float, 16> v3234_data(0.0f);
              v3234_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[11]), v8_g);
              if (v8_g) {
                (v3234_data + (v3110_data * v3232_data)).copy_to(ir1 + (11));
              }
              #pragma unroll
              for (int32_t v3238_n1 = 0; v3238_n1 < 12; ++v3238_n1) {
                int32_t v3239_a = 0 + v3238_n1;
                tensorforge::intel_esimd::simd<float, 16> v3241_data(0.0f);
                v3241_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[v3238_n1]), v8_g);
                if (v8_g) {
                  v3241_data.copy_to(r1 + (v3238_n1));
                }
              }
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v3245_i1 = 0; v3245_i1 < 12; ++v3245_i1) {
                int32_t v3246_a = 0 + v3245_i1;
                tensorforge::intel_esimd::simd<float, 16> v3248_data(0.0f);
                v3248_data.merge(tensorforge::intel_esimd::simd<float, 16>(r1[v3245_i1]), v8_g);
                int32_t v3253_a = 6_i32 + (v3245_i1 * 12);
                if (v8_g) {
                  s1[v3253_a] = v3248_data;
                }
              }
              float r2[12]{};
              // r2 = +(glb_m4 * s1) + None
              // [(0, 12), (0, 12)] [(0, 12)]
              float ir2[12]{};
              tensorforge::intel_esimd::simd_mask<16> v3257_g = v7_lead < 12;
              int32_t v3260_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3264_data(0.0f);
              v3264_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m4[0_i32]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3265_data(0.0f);
              v3265_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[0]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3267_data(0.0f);
              v3267_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v3257_g);
              if (v3257_g) {
                (v3267_data + (v3264_data * v3265_data)).copy_to(ir2 + (0));
              }
              int32_t v3271_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3276_data(0.0f);
              v3276_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[12]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3278_data(0.0f);
              v3278_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v3257_g);
              if (v3257_g) {
                (v3278_data + (v3264_data * v3276_data)).copy_to(ir2 + (1));
              }
              int32_t v3282_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3287_data(0.0f);
              v3287_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[24]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3289_data(0.0f);
              v3289_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v3257_g);
              if (v3257_g) {
                (v3289_data + (v3264_data * v3287_data)).copy_to(ir2 + (2));
              }
              int32_t v3293_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3298_data(0.0f);
              v3298_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[36]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3300_data(0.0f);
              v3300_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v3257_g);
              if (v3257_g) {
                (v3300_data + (v3264_data * v3298_data)).copy_to(ir2 + (3));
              }
              int32_t v3304_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3309_data(0.0f);
              v3309_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[48]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3311_data(0.0f);
              v3311_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v3257_g);
              if (v3257_g) {
                (v3311_data + (v3264_data * v3309_data)).copy_to(ir2 + (4));
              }
              int32_t v3315_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3320_data(0.0f);
              v3320_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[60]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3322_data(0.0f);
              v3322_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v3257_g);
              if (v3257_g) {
                (v3322_data + (v3264_data * v3320_data)).copy_to(ir2 + (5));
              }
              int32_t v3326_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3331_data(0.0f);
              v3331_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[72]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3333_data(0.0f);
              v3333_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v3257_g);
              if (v3257_g) {
                (v3333_data + (v3264_data * v3331_data)).copy_to(ir2 + (6));
              }
              int32_t v3337_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3342_data(0.0f);
              v3342_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[84]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3344_data(0.0f);
              v3344_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v3257_g);
              if (v3257_g) {
                (v3344_data + (v3264_data * v3342_data)).copy_to(ir2 + (7));
              }
              int32_t v3348_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3353_data(0.0f);
              v3353_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[96]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3355_data(0.0f);
              v3355_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[8]), v3257_g);
              if (v3257_g) {
                (v3355_data + (v3264_data * v3353_data)).copy_to(ir2 + (8));
              }
              int32_t v3359_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3364_data(0.0f);
              v3364_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[108]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3366_data(0.0f);
              v3366_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[9]), v3257_g);
              if (v3257_g) {
                (v3366_data + (v3264_data * v3364_data)).copy_to(ir2 + (9));
              }
              int32_t v3370_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3375_data(0.0f);
              v3375_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[120]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3377_data(0.0f);
              v3377_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[10]), v3257_g);
              if (v3257_g) {
                (v3377_data + (v3264_data * v3375_data)).copy_to(ir2 + (10));
              }
              int32_t v3381_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3386_data(0.0f);
              v3386_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[132]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3388_data(0.0f);
              v3388_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[11]), v3257_g);
              if (v3257_g) {
                (v3388_data + (v3264_data * v3386_data)).copy_to(ir2 + (11));
              }
              int32_t v3394_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3398_data(0.0f);
              v3398_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m4[12_i32]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3399_data(0.0f);
              v3399_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[1]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3401_data(0.0f);
              v3401_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v3257_g);
              if (v3257_g) {
                (v3401_data + (v3398_data * v3399_data)).copy_to(ir2 + (0));
              }
              int32_t v3405_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3410_data(0.0f);
              v3410_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[13]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3412_data(0.0f);
              v3412_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v3257_g);
              if (v3257_g) {
                (v3412_data + (v3398_data * v3410_data)).copy_to(ir2 + (1));
              }
              int32_t v3416_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3421_data(0.0f);
              v3421_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[25]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3423_data(0.0f);
              v3423_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v3257_g);
              if (v3257_g) {
                (v3423_data + (v3398_data * v3421_data)).copy_to(ir2 + (2));
              }
              int32_t v3427_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3432_data(0.0f);
              v3432_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[37]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3434_data(0.0f);
              v3434_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v3257_g);
              if (v3257_g) {
                (v3434_data + (v3398_data * v3432_data)).copy_to(ir2 + (3));
              }
              int32_t v3438_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3443_data(0.0f);
              v3443_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[49]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3445_data(0.0f);
              v3445_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v3257_g);
              if (v3257_g) {
                (v3445_data + (v3398_data * v3443_data)).copy_to(ir2 + (4));
              }
              int32_t v3449_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3454_data(0.0f);
              v3454_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[61]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3456_data(0.0f);
              v3456_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v3257_g);
              if (v3257_g) {
                (v3456_data + (v3398_data * v3454_data)).copy_to(ir2 + (5));
              }
              int32_t v3460_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3465_data(0.0f);
              v3465_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[73]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3467_data(0.0f);
              v3467_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v3257_g);
              if (v3257_g) {
                (v3467_data + (v3398_data * v3465_data)).copy_to(ir2 + (6));
              }
              int32_t v3471_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3476_data(0.0f);
              v3476_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[85]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3478_data(0.0f);
              v3478_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v3257_g);
              if (v3257_g) {
                (v3478_data + (v3398_data * v3476_data)).copy_to(ir2 + (7));
              }
              int32_t v3482_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3487_data(0.0f);
              v3487_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[97]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3489_data(0.0f);
              v3489_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[8]), v3257_g);
              if (v3257_g) {
                (v3489_data + (v3398_data * v3487_data)).copy_to(ir2 + (8));
              }
              int32_t v3493_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3498_data(0.0f);
              v3498_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[109]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3500_data(0.0f);
              v3500_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[9]), v3257_g);
              if (v3257_g) {
                (v3500_data + (v3398_data * v3498_data)).copy_to(ir2 + (9));
              }
              int32_t v3504_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3509_data(0.0f);
              v3509_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[121]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3511_data(0.0f);
              v3511_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[10]), v3257_g);
              if (v3257_g) {
                (v3511_data + (v3398_data * v3509_data)).copy_to(ir2 + (10));
              }
              int32_t v3515_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3520_data(0.0f);
              v3520_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[133]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3522_data(0.0f);
              v3522_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[11]), v3257_g);
              if (v3257_g) {
                (v3522_data + (v3398_data * v3520_data)).copy_to(ir2 + (11));
              }
              int32_t v3528_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3532_data(0.0f);
              v3532_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m4[24_i32]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3533_data(0.0f);
              v3533_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[2]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3535_data(0.0f);
              v3535_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v3257_g);
              if (v3257_g) {
                (v3535_data + (v3532_data * v3533_data)).copy_to(ir2 + (0));
              }
              int32_t v3539_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3544_data(0.0f);
              v3544_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[14]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3546_data(0.0f);
              v3546_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v3257_g);
              if (v3257_g) {
                (v3546_data + (v3532_data * v3544_data)).copy_to(ir2 + (1));
              }
              int32_t v3550_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3555_data(0.0f);
              v3555_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[26]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3557_data(0.0f);
              v3557_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v3257_g);
              if (v3257_g) {
                (v3557_data + (v3532_data * v3555_data)).copy_to(ir2 + (2));
              }
              int32_t v3561_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3566_data(0.0f);
              v3566_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[38]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3568_data(0.0f);
              v3568_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v3257_g);
              if (v3257_g) {
                (v3568_data + (v3532_data * v3566_data)).copy_to(ir2 + (3));
              }
              int32_t v3572_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3577_data(0.0f);
              v3577_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[50]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3579_data(0.0f);
              v3579_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v3257_g);
              if (v3257_g) {
                (v3579_data + (v3532_data * v3577_data)).copy_to(ir2 + (4));
              }
              int32_t v3583_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3588_data(0.0f);
              v3588_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[62]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3590_data(0.0f);
              v3590_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v3257_g);
              if (v3257_g) {
                (v3590_data + (v3532_data * v3588_data)).copy_to(ir2 + (5));
              }
              int32_t v3594_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3599_data(0.0f);
              v3599_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[74]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3601_data(0.0f);
              v3601_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v3257_g);
              if (v3257_g) {
                (v3601_data + (v3532_data * v3599_data)).copy_to(ir2 + (6));
              }
              int32_t v3605_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3610_data(0.0f);
              v3610_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[86]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3612_data(0.0f);
              v3612_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v3257_g);
              if (v3257_g) {
                (v3612_data + (v3532_data * v3610_data)).copy_to(ir2 + (7));
              }
              int32_t v3616_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3621_data(0.0f);
              v3621_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[98]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3623_data(0.0f);
              v3623_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[8]), v3257_g);
              if (v3257_g) {
                (v3623_data + (v3532_data * v3621_data)).copy_to(ir2 + (8));
              }
              int32_t v3627_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3632_data(0.0f);
              v3632_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[110]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3634_data(0.0f);
              v3634_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[9]), v3257_g);
              if (v3257_g) {
                (v3634_data + (v3532_data * v3632_data)).copy_to(ir2 + (9));
              }
              int32_t v3638_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3643_data(0.0f);
              v3643_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[122]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3645_data(0.0f);
              v3645_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[10]), v3257_g);
              if (v3257_g) {
                (v3645_data + (v3532_data * v3643_data)).copy_to(ir2 + (10));
              }
              int32_t v3649_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3654_data(0.0f);
              v3654_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[134]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3656_data(0.0f);
              v3656_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[11]), v3257_g);
              if (v3257_g) {
                (v3656_data + (v3532_data * v3654_data)).copy_to(ir2 + (11));
              }
              int32_t v3662_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3666_data(0.0f);
              v3666_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m4[36_i32]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3667_data(0.0f);
              v3667_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[3]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3669_data(0.0f);
              v3669_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v3257_g);
              if (v3257_g) {
                (v3669_data + (v3666_data * v3667_data)).copy_to(ir2 + (0));
              }
              int32_t v3673_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3678_data(0.0f);
              v3678_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[15]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3680_data(0.0f);
              v3680_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v3257_g);
              if (v3257_g) {
                (v3680_data + (v3666_data * v3678_data)).copy_to(ir2 + (1));
              }
              int32_t v3684_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3689_data(0.0f);
              v3689_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[27]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3691_data(0.0f);
              v3691_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v3257_g);
              if (v3257_g) {
                (v3691_data + (v3666_data * v3689_data)).copy_to(ir2 + (2));
              }
              int32_t v3695_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3700_data(0.0f);
              v3700_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[39]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3702_data(0.0f);
              v3702_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v3257_g);
              if (v3257_g) {
                (v3702_data + (v3666_data * v3700_data)).copy_to(ir2 + (3));
              }
              int32_t v3706_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3711_data(0.0f);
              v3711_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[51]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3713_data(0.0f);
              v3713_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v3257_g);
              if (v3257_g) {
                (v3713_data + (v3666_data * v3711_data)).copy_to(ir2 + (4));
              }
              int32_t v3717_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3722_data(0.0f);
              v3722_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[63]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3724_data(0.0f);
              v3724_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v3257_g);
              if (v3257_g) {
                (v3724_data + (v3666_data * v3722_data)).copy_to(ir2 + (5));
              }
              int32_t v3728_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3733_data(0.0f);
              v3733_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[75]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3735_data(0.0f);
              v3735_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v3257_g);
              if (v3257_g) {
                (v3735_data + (v3666_data * v3733_data)).copy_to(ir2 + (6));
              }
              int32_t v3739_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3744_data(0.0f);
              v3744_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[87]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3746_data(0.0f);
              v3746_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v3257_g);
              if (v3257_g) {
                (v3746_data + (v3666_data * v3744_data)).copy_to(ir2 + (7));
              }
              int32_t v3750_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3755_data(0.0f);
              v3755_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[99]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3757_data(0.0f);
              v3757_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[8]), v3257_g);
              if (v3257_g) {
                (v3757_data + (v3666_data * v3755_data)).copy_to(ir2 + (8));
              }
              int32_t v3761_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3766_data(0.0f);
              v3766_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[111]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3768_data(0.0f);
              v3768_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[9]), v3257_g);
              if (v3257_g) {
                (v3768_data + (v3666_data * v3766_data)).copy_to(ir2 + (9));
              }
              int32_t v3772_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3777_data(0.0f);
              v3777_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[123]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3779_data(0.0f);
              v3779_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[10]), v3257_g);
              if (v3257_g) {
                (v3779_data + (v3666_data * v3777_data)).copy_to(ir2 + (10));
              }
              int32_t v3783_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3788_data(0.0f);
              v3788_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[135]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3790_data(0.0f);
              v3790_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[11]), v3257_g);
              if (v3257_g) {
                (v3790_data + (v3666_data * v3788_data)).copy_to(ir2 + (11));
              }
              int32_t v3796_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3800_data(0.0f);
              v3800_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m4[48_i32]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3801_data(0.0f);
              v3801_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[4]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3803_data(0.0f);
              v3803_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v3257_g);
              if (v3257_g) {
                (v3803_data + (v3800_data * v3801_data)).copy_to(ir2 + (0));
              }
              int32_t v3807_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3812_data(0.0f);
              v3812_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[16]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3814_data(0.0f);
              v3814_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v3257_g);
              if (v3257_g) {
                (v3814_data + (v3800_data * v3812_data)).copy_to(ir2 + (1));
              }
              int32_t v3818_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3823_data(0.0f);
              v3823_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[28]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3825_data(0.0f);
              v3825_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v3257_g);
              if (v3257_g) {
                (v3825_data + (v3800_data * v3823_data)).copy_to(ir2 + (2));
              }
              int32_t v3829_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3834_data(0.0f);
              v3834_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[40]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3836_data(0.0f);
              v3836_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v3257_g);
              if (v3257_g) {
                (v3836_data + (v3800_data * v3834_data)).copy_to(ir2 + (3));
              }
              int32_t v3840_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3845_data(0.0f);
              v3845_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[52]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3847_data(0.0f);
              v3847_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v3257_g);
              if (v3257_g) {
                (v3847_data + (v3800_data * v3845_data)).copy_to(ir2 + (4));
              }
              int32_t v3851_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3856_data(0.0f);
              v3856_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[64]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3858_data(0.0f);
              v3858_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v3257_g);
              if (v3257_g) {
                (v3858_data + (v3800_data * v3856_data)).copy_to(ir2 + (5));
              }
              int32_t v3862_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3867_data(0.0f);
              v3867_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[76]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3869_data(0.0f);
              v3869_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v3257_g);
              if (v3257_g) {
                (v3869_data + (v3800_data * v3867_data)).copy_to(ir2 + (6));
              }
              int32_t v3873_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3878_data(0.0f);
              v3878_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[88]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3880_data(0.0f);
              v3880_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v3257_g);
              if (v3257_g) {
                (v3880_data + (v3800_data * v3878_data)).copy_to(ir2 + (7));
              }
              int32_t v3884_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3889_data(0.0f);
              v3889_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[100]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3891_data(0.0f);
              v3891_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[8]), v3257_g);
              if (v3257_g) {
                (v3891_data + (v3800_data * v3889_data)).copy_to(ir2 + (8));
              }
              int32_t v3895_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3900_data(0.0f);
              v3900_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[112]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3902_data(0.0f);
              v3902_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[9]), v3257_g);
              if (v3257_g) {
                (v3902_data + (v3800_data * v3900_data)).copy_to(ir2 + (9));
              }
              int32_t v3906_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3911_data(0.0f);
              v3911_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[124]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3913_data(0.0f);
              v3913_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[10]), v3257_g);
              if (v3257_g) {
                (v3913_data + (v3800_data * v3911_data)).copy_to(ir2 + (10));
              }
              int32_t v3917_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3922_data(0.0f);
              v3922_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[136]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3924_data(0.0f);
              v3924_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[11]), v3257_g);
              if (v3257_g) {
                (v3924_data + (v3800_data * v3922_data)).copy_to(ir2 + (11));
              }
              int32_t v3930_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3934_data(0.0f);
              v3934_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m4[60_i32]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3935_data(0.0f);
              v3935_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[5]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3937_data(0.0f);
              v3937_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v3257_g);
              if (v3257_g) {
                (v3937_data + (v3934_data * v3935_data)).copy_to(ir2 + (0));
              }
              int32_t v3941_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3946_data(0.0f);
              v3946_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[17]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3948_data(0.0f);
              v3948_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v3257_g);
              if (v3257_g) {
                (v3948_data + (v3934_data * v3946_data)).copy_to(ir2 + (1));
              }
              int32_t v3952_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3957_data(0.0f);
              v3957_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[29]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3959_data(0.0f);
              v3959_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v3257_g);
              if (v3257_g) {
                (v3959_data + (v3934_data * v3957_data)).copy_to(ir2 + (2));
              }
              int32_t v3963_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3968_data(0.0f);
              v3968_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[41]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3970_data(0.0f);
              v3970_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v3257_g);
              if (v3257_g) {
                (v3970_data + (v3934_data * v3968_data)).copy_to(ir2 + (3));
              }
              int32_t v3974_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3979_data(0.0f);
              v3979_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[53]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3981_data(0.0f);
              v3981_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v3257_g);
              if (v3257_g) {
                (v3981_data + (v3934_data * v3979_data)).copy_to(ir2 + (4));
              }
              int32_t v3985_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3990_data(0.0f);
              v3990_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[65]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v3992_data(0.0f);
              v3992_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v3257_g);
              if (v3257_g) {
                (v3992_data + (v3934_data * v3990_data)).copy_to(ir2 + (5));
              }
              int32_t v3996_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v4001_data(0.0f);
              v4001_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[77]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4003_data(0.0f);
              v4003_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v3257_g);
              if (v3257_g) {
                (v4003_data + (v3934_data * v4001_data)).copy_to(ir2 + (6));
              }
              int32_t v4007_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v4012_data(0.0f);
              v4012_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[89]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4014_data(0.0f);
              v4014_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v3257_g);
              if (v3257_g) {
                (v4014_data + (v3934_data * v4012_data)).copy_to(ir2 + (7));
              }
              int32_t v4018_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v4023_data(0.0f);
              v4023_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[101]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4025_data(0.0f);
              v4025_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[8]), v3257_g);
              if (v3257_g) {
                (v4025_data + (v3934_data * v4023_data)).copy_to(ir2 + (8));
              }
              int32_t v4029_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v4034_data(0.0f);
              v4034_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[113]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4036_data(0.0f);
              v4036_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[9]), v3257_g);
              if (v3257_g) {
                (v4036_data + (v3934_data * v4034_data)).copy_to(ir2 + (9));
              }
              int32_t v4040_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v4045_data(0.0f);
              v4045_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[125]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4047_data(0.0f);
              v4047_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[10]), v3257_g);
              if (v3257_g) {
                (v4047_data + (v3934_data * v4045_data)).copy_to(ir2 + (10));
              }
              int32_t v4051_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v4056_data(0.0f);
              v4056_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[137]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4058_data(0.0f);
              v4058_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[11]), v3257_g);
              if (v3257_g) {
                (v4058_data + (v3934_data * v4056_data)).copy_to(ir2 + (11));
              }
              int32_t v4064_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v4068_data(0.0f);
              v4068_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m4[72_i32]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4069_data(0.0f);
              v4069_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[6]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4071_data(0.0f);
              v4071_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v3257_g);
              if (v3257_g) {
                (v4071_data + (v4068_data * v4069_data)).copy_to(ir2 + (0));
              }
              int32_t v4075_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v4080_data(0.0f);
              v4080_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[18]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4082_data(0.0f);
              v4082_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v3257_g);
              if (v3257_g) {
                (v4082_data + (v4068_data * v4080_data)).copy_to(ir2 + (1));
              }
              int32_t v4086_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v4091_data(0.0f);
              v4091_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[30]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4093_data(0.0f);
              v4093_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v3257_g);
              if (v3257_g) {
                (v4093_data + (v4068_data * v4091_data)).copy_to(ir2 + (2));
              }
              int32_t v4097_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v4102_data(0.0f);
              v4102_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[42]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4104_data(0.0f);
              v4104_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v3257_g);
              if (v3257_g) {
                (v4104_data + (v4068_data * v4102_data)).copy_to(ir2 + (3));
              }
              int32_t v4108_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v4113_data(0.0f);
              v4113_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[54]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4115_data(0.0f);
              v4115_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v3257_g);
              if (v3257_g) {
                (v4115_data + (v4068_data * v4113_data)).copy_to(ir2 + (4));
              }
              int32_t v4119_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v4124_data(0.0f);
              v4124_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[66]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4126_data(0.0f);
              v4126_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v3257_g);
              if (v3257_g) {
                (v4126_data + (v4068_data * v4124_data)).copy_to(ir2 + (5));
              }
              int32_t v4130_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v4135_data(0.0f);
              v4135_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[78]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4137_data(0.0f);
              v4137_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v3257_g);
              if (v3257_g) {
                (v4137_data + (v4068_data * v4135_data)).copy_to(ir2 + (6));
              }
              int32_t v4141_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v4146_data(0.0f);
              v4146_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[90]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4148_data(0.0f);
              v4148_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v3257_g);
              if (v3257_g) {
                (v4148_data + (v4068_data * v4146_data)).copy_to(ir2 + (7));
              }
              int32_t v4152_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v4157_data(0.0f);
              v4157_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[102]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4159_data(0.0f);
              v4159_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[8]), v3257_g);
              if (v3257_g) {
                (v4159_data + (v4068_data * v4157_data)).copy_to(ir2 + (8));
              }
              int32_t v4163_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v4168_data(0.0f);
              v4168_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[114]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4170_data(0.0f);
              v4170_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[9]), v3257_g);
              if (v3257_g) {
                (v4170_data + (v4068_data * v4168_data)).copy_to(ir2 + (9));
              }
              int32_t v4174_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v4179_data(0.0f);
              v4179_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[126]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4181_data(0.0f);
              v4181_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[10]), v3257_g);
              if (v3257_g) {
                (v4181_data + (v4068_data * v4179_data)).copy_to(ir2 + (10));
              }
              int32_t v4185_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v4190_data(0.0f);
              v4190_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[138]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4192_data(0.0f);
              v4192_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[11]), v3257_g);
              if (v3257_g) {
                (v4192_data + (v4068_data * v4190_data)).copy_to(ir2 + (11));
              }
              int32_t v4198_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4202_data(0.0f);
              v4202_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m4[84_i32]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4203_data(0.0f);
              v4203_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[7]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4205_data(0.0f);
              v4205_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v3257_g);
              if (v3257_g) {
                (v4205_data + (v4202_data * v4203_data)).copy_to(ir2 + (0));
              }
              int32_t v4209_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4214_data(0.0f);
              v4214_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[19]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4216_data(0.0f);
              v4216_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v3257_g);
              if (v3257_g) {
                (v4216_data + (v4202_data * v4214_data)).copy_to(ir2 + (1));
              }
              int32_t v4220_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4225_data(0.0f);
              v4225_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[31]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4227_data(0.0f);
              v4227_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v3257_g);
              if (v3257_g) {
                (v4227_data + (v4202_data * v4225_data)).copy_to(ir2 + (2));
              }
              int32_t v4231_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4236_data(0.0f);
              v4236_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[43]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4238_data(0.0f);
              v4238_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v3257_g);
              if (v3257_g) {
                (v4238_data + (v4202_data * v4236_data)).copy_to(ir2 + (3));
              }
              int32_t v4242_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4247_data(0.0f);
              v4247_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[55]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4249_data(0.0f);
              v4249_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v3257_g);
              if (v3257_g) {
                (v4249_data + (v4202_data * v4247_data)).copy_to(ir2 + (4));
              }
              int32_t v4253_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4258_data(0.0f);
              v4258_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[67]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4260_data(0.0f);
              v4260_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v3257_g);
              if (v3257_g) {
                (v4260_data + (v4202_data * v4258_data)).copy_to(ir2 + (5));
              }
              int32_t v4264_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4269_data(0.0f);
              v4269_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[79]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4271_data(0.0f);
              v4271_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v3257_g);
              if (v3257_g) {
                (v4271_data + (v4202_data * v4269_data)).copy_to(ir2 + (6));
              }
              int32_t v4275_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4280_data(0.0f);
              v4280_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[91]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4282_data(0.0f);
              v4282_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v3257_g);
              if (v3257_g) {
                (v4282_data + (v4202_data * v4280_data)).copy_to(ir2 + (7));
              }
              int32_t v4286_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4291_data(0.0f);
              v4291_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[103]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4293_data(0.0f);
              v4293_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[8]), v3257_g);
              if (v3257_g) {
                (v4293_data + (v4202_data * v4291_data)).copy_to(ir2 + (8));
              }
              int32_t v4297_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4302_data(0.0f);
              v4302_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[115]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4304_data(0.0f);
              v4304_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[9]), v3257_g);
              if (v3257_g) {
                (v4304_data + (v4202_data * v4302_data)).copy_to(ir2 + (9));
              }
              int32_t v4308_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4313_data(0.0f);
              v4313_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[127]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4315_data(0.0f);
              v4315_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[10]), v3257_g);
              if (v3257_g) {
                (v4315_data + (v4202_data * v4313_data)).copy_to(ir2 + (10));
              }
              int32_t v4319_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4324_data(0.0f);
              v4324_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[139]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4326_data(0.0f);
              v4326_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[11]), v3257_g);
              if (v3257_g) {
                (v4326_data + (v4202_data * v4324_data)).copy_to(ir2 + (11));
              }
              int32_t v4332_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4336_data(0.0f);
              v4336_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m4[96_i32]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4337_data(0.0f);
              v4337_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[8]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4339_data(0.0f);
              v4339_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v3257_g);
              if (v3257_g) {
                (v4339_data + (v4336_data * v4337_data)).copy_to(ir2 + (0));
              }
              int32_t v4343_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4348_data(0.0f);
              v4348_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[20]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4350_data(0.0f);
              v4350_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v3257_g);
              if (v3257_g) {
                (v4350_data + (v4336_data * v4348_data)).copy_to(ir2 + (1));
              }
              int32_t v4354_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4359_data(0.0f);
              v4359_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[32]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4361_data(0.0f);
              v4361_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v3257_g);
              if (v3257_g) {
                (v4361_data + (v4336_data * v4359_data)).copy_to(ir2 + (2));
              }
              int32_t v4365_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4370_data(0.0f);
              v4370_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[44]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4372_data(0.0f);
              v4372_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v3257_g);
              if (v3257_g) {
                (v4372_data + (v4336_data * v4370_data)).copy_to(ir2 + (3));
              }
              int32_t v4376_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4381_data(0.0f);
              v4381_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[56]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4383_data(0.0f);
              v4383_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v3257_g);
              if (v3257_g) {
                (v4383_data + (v4336_data * v4381_data)).copy_to(ir2 + (4));
              }
              int32_t v4387_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4392_data(0.0f);
              v4392_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[68]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4394_data(0.0f);
              v4394_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v3257_g);
              if (v3257_g) {
                (v4394_data + (v4336_data * v4392_data)).copy_to(ir2 + (5));
              }
              int32_t v4398_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4403_data(0.0f);
              v4403_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[80]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4405_data(0.0f);
              v4405_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v3257_g);
              if (v3257_g) {
                (v4405_data + (v4336_data * v4403_data)).copy_to(ir2 + (6));
              }
              int32_t v4409_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4414_data(0.0f);
              v4414_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[92]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4416_data(0.0f);
              v4416_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v3257_g);
              if (v3257_g) {
                (v4416_data + (v4336_data * v4414_data)).copy_to(ir2 + (7));
              }
              int32_t v4420_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4425_data(0.0f);
              v4425_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[104]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4427_data(0.0f);
              v4427_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[8]), v3257_g);
              if (v3257_g) {
                (v4427_data + (v4336_data * v4425_data)).copy_to(ir2 + (8));
              }
              int32_t v4431_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4436_data(0.0f);
              v4436_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[116]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4438_data(0.0f);
              v4438_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[9]), v3257_g);
              if (v3257_g) {
                (v4438_data + (v4336_data * v4436_data)).copy_to(ir2 + (9));
              }
              int32_t v4442_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4447_data(0.0f);
              v4447_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[128]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4449_data(0.0f);
              v4449_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[10]), v3257_g);
              if (v3257_g) {
                (v4449_data + (v4336_data * v4447_data)).copy_to(ir2 + (10));
              }
              int32_t v4453_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4458_data(0.0f);
              v4458_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[140]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4460_data(0.0f);
              v4460_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[11]), v3257_g);
              if (v3257_g) {
                (v4460_data + (v4336_data * v4458_data)).copy_to(ir2 + (11));
              }
              int32_t v4466_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4470_data(0.0f);
              v4470_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m4[108_i32]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4471_data(0.0f);
              v4471_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[9]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4473_data(0.0f);
              v4473_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v3257_g);
              if (v3257_g) {
                (v4473_data + (v4470_data * v4471_data)).copy_to(ir2 + (0));
              }
              int32_t v4477_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4482_data(0.0f);
              v4482_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[21]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4484_data(0.0f);
              v4484_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v3257_g);
              if (v3257_g) {
                (v4484_data + (v4470_data * v4482_data)).copy_to(ir2 + (1));
              }
              int32_t v4488_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4493_data(0.0f);
              v4493_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[33]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4495_data(0.0f);
              v4495_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v3257_g);
              if (v3257_g) {
                (v4495_data + (v4470_data * v4493_data)).copy_to(ir2 + (2));
              }
              int32_t v4499_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4504_data(0.0f);
              v4504_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[45]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4506_data(0.0f);
              v4506_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v3257_g);
              if (v3257_g) {
                (v4506_data + (v4470_data * v4504_data)).copy_to(ir2 + (3));
              }
              int32_t v4510_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4515_data(0.0f);
              v4515_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[57]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4517_data(0.0f);
              v4517_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v3257_g);
              if (v3257_g) {
                (v4517_data + (v4470_data * v4515_data)).copy_to(ir2 + (4));
              }
              int32_t v4521_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4526_data(0.0f);
              v4526_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[69]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4528_data(0.0f);
              v4528_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v3257_g);
              if (v3257_g) {
                (v4528_data + (v4470_data * v4526_data)).copy_to(ir2 + (5));
              }
              int32_t v4532_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4537_data(0.0f);
              v4537_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[81]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4539_data(0.0f);
              v4539_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v3257_g);
              if (v3257_g) {
                (v4539_data + (v4470_data * v4537_data)).copy_to(ir2 + (6));
              }
              int32_t v4543_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4548_data(0.0f);
              v4548_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[93]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4550_data(0.0f);
              v4550_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v3257_g);
              if (v3257_g) {
                (v4550_data + (v4470_data * v4548_data)).copy_to(ir2 + (7));
              }
              int32_t v4554_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4559_data(0.0f);
              v4559_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[105]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4561_data(0.0f);
              v4561_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[8]), v3257_g);
              if (v3257_g) {
                (v4561_data + (v4470_data * v4559_data)).copy_to(ir2 + (8));
              }
              int32_t v4565_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4570_data(0.0f);
              v4570_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[117]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4572_data(0.0f);
              v4572_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[9]), v3257_g);
              if (v3257_g) {
                (v4572_data + (v4470_data * v4570_data)).copy_to(ir2 + (9));
              }
              int32_t v4576_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4581_data(0.0f);
              v4581_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[129]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4583_data(0.0f);
              v4583_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[10]), v3257_g);
              if (v3257_g) {
                (v4583_data + (v4470_data * v4581_data)).copy_to(ir2 + (10));
              }
              int32_t v4587_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4592_data(0.0f);
              v4592_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[141]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4594_data(0.0f);
              v4594_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[11]), v3257_g);
              if (v3257_g) {
                (v4594_data + (v4470_data * v4592_data)).copy_to(ir2 + (11));
              }
              int32_t v4600_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4604_data(0.0f);
              v4604_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m4[120_i32]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4605_data(0.0f);
              v4605_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[10]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4607_data(0.0f);
              v4607_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v3257_g);
              if (v3257_g) {
                (v4607_data + (v4604_data * v4605_data)).copy_to(ir2 + (0));
              }
              int32_t v4611_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4616_data(0.0f);
              v4616_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[22]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4618_data(0.0f);
              v4618_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v3257_g);
              if (v3257_g) {
                (v4618_data + (v4604_data * v4616_data)).copy_to(ir2 + (1));
              }
              int32_t v4622_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4627_data(0.0f);
              v4627_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[34]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4629_data(0.0f);
              v4629_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v3257_g);
              if (v3257_g) {
                (v4629_data + (v4604_data * v4627_data)).copy_to(ir2 + (2));
              }
              int32_t v4633_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4638_data(0.0f);
              v4638_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[46]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4640_data(0.0f);
              v4640_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v3257_g);
              if (v3257_g) {
                (v4640_data + (v4604_data * v4638_data)).copy_to(ir2 + (3));
              }
              int32_t v4644_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4649_data(0.0f);
              v4649_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[58]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4651_data(0.0f);
              v4651_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v3257_g);
              if (v3257_g) {
                (v4651_data + (v4604_data * v4649_data)).copy_to(ir2 + (4));
              }
              int32_t v4655_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4660_data(0.0f);
              v4660_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[70]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4662_data(0.0f);
              v4662_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v3257_g);
              if (v3257_g) {
                (v4662_data + (v4604_data * v4660_data)).copy_to(ir2 + (5));
              }
              int32_t v4666_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4671_data(0.0f);
              v4671_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[82]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4673_data(0.0f);
              v4673_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v3257_g);
              if (v3257_g) {
                (v4673_data + (v4604_data * v4671_data)).copy_to(ir2 + (6));
              }
              int32_t v4677_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4682_data(0.0f);
              v4682_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[94]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4684_data(0.0f);
              v4684_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v3257_g);
              if (v3257_g) {
                (v4684_data + (v4604_data * v4682_data)).copy_to(ir2 + (7));
              }
              int32_t v4688_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4693_data(0.0f);
              v4693_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[106]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4695_data(0.0f);
              v4695_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[8]), v3257_g);
              if (v3257_g) {
                (v4695_data + (v4604_data * v4693_data)).copy_to(ir2 + (8));
              }
              int32_t v4699_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4704_data(0.0f);
              v4704_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[118]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4706_data(0.0f);
              v4706_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[9]), v3257_g);
              if (v3257_g) {
                (v4706_data + (v4604_data * v4704_data)).copy_to(ir2 + (9));
              }
              int32_t v4710_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4715_data(0.0f);
              v4715_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[130]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4717_data(0.0f);
              v4717_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[10]), v3257_g);
              if (v3257_g) {
                (v4717_data + (v4604_data * v4715_data)).copy_to(ir2 + (10));
              }
              int32_t v4721_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4726_data(0.0f);
              v4726_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[142]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4728_data(0.0f);
              v4728_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[11]), v3257_g);
              if (v3257_g) {
                (v4728_data + (v4604_data * v4726_data)).copy_to(ir2 + (11));
              }
              int32_t v4734_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4738_data(0.0f);
              v4738_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m4[132_i32]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4739_data(0.0f);
              v4739_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[11]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4741_data(0.0f);
              v4741_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v3257_g);
              if (v3257_g) {
                (v4741_data + (v4738_data * v4739_data)).copy_to(ir2 + (0));
              }
              int32_t v4745_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4750_data(0.0f);
              v4750_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[23]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4752_data(0.0f);
              v4752_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v3257_g);
              if (v3257_g) {
                (v4752_data + (v4738_data * v4750_data)).copy_to(ir2 + (1));
              }
              int32_t v4756_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4761_data(0.0f);
              v4761_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[35]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4763_data(0.0f);
              v4763_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v3257_g);
              if (v3257_g) {
                (v4763_data + (v4738_data * v4761_data)).copy_to(ir2 + (2));
              }
              int32_t v4767_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4772_data(0.0f);
              v4772_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[47]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4774_data(0.0f);
              v4774_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v3257_g);
              if (v3257_g) {
                (v4774_data + (v4738_data * v4772_data)).copy_to(ir2 + (3));
              }
              int32_t v4778_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4783_data(0.0f);
              v4783_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[59]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4785_data(0.0f);
              v4785_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v3257_g);
              if (v3257_g) {
                (v4785_data + (v4738_data * v4783_data)).copy_to(ir2 + (4));
              }
              int32_t v4789_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4794_data(0.0f);
              v4794_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[71]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4796_data(0.0f);
              v4796_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v3257_g);
              if (v3257_g) {
                (v4796_data + (v4738_data * v4794_data)).copy_to(ir2 + (5));
              }
              int32_t v4800_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4805_data(0.0f);
              v4805_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[83]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4807_data(0.0f);
              v4807_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v3257_g);
              if (v3257_g) {
                (v4807_data + (v4738_data * v4805_data)).copy_to(ir2 + (6));
              }
              int32_t v4811_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4816_data(0.0f);
              v4816_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[95]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4818_data(0.0f);
              v4818_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v3257_g);
              if (v3257_g) {
                (v4818_data + (v4738_data * v4816_data)).copy_to(ir2 + (7));
              }
              int32_t v4822_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4827_data(0.0f);
              v4827_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[107]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4829_data(0.0f);
              v4829_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[8]), v3257_g);
              if (v3257_g) {
                (v4829_data + (v4738_data * v4827_data)).copy_to(ir2 + (8));
              }
              int32_t v4833_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4838_data(0.0f);
              v4838_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[119]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4840_data(0.0f);
              v4840_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[9]), v3257_g);
              if (v3257_g) {
                (v4840_data + (v4738_data * v4838_data)).copy_to(ir2 + (9));
              }
              int32_t v4844_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4849_data(0.0f);
              v4849_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[131]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4851_data(0.0f);
              v4851_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[10]), v3257_g);
              if (v3257_g) {
                (v4851_data + (v4738_data * v4849_data)).copy_to(ir2 + (10));
              }
              int32_t v4855_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4860_data(0.0f);
              v4860_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[143]), v3257_g);
              tensorforge::intel_esimd::simd<float, 16> v4862_data(0.0f);
              v4862_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[11]), v3257_g);
              if (v3257_g) {
                (v4862_data + (v4738_data * v4860_data)).copy_to(ir2 + (11));
              }
              #pragma unroll
              for (int32_t v4866_n1 = 0; v4866_n1 < 12; ++v4866_n1) {
                int32_t v4867_a = 0 + v4866_n1;
                tensorforge::intel_esimd::simd<float, 16> v4869_data(0.0f);
                v4869_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[v4866_n1]), v3257_g);
                if (v3257_g) {
                  v4869_data.copy_to(r2 + (v4866_n1));
                }
              }
              // glb_m3 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v4873_i1 = 0; v4873_i1 < 12; ++v4873_i1) {
                int32_t v4874_a = 0 + v4873_i1;
                tensorforge::intel_esimd::simd<float, 16> v4876_data(0.0f);
                v4876_data.merge(tensorforge::intel_esimd::simd<float, 16>(r2[v4873_i1]), v3257_g);
                if (v3257_g) {
                  v4876_data.copy_to(glb_m3 + ((v4873_i1 * 12)));
                }
              }
            }
          }
        }
      });
    }
  });
}

