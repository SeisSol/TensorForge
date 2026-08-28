// === base name ===
kernel_e7f2438624

// === header ===
void launcher_kernel_e7f2438624(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_e7f2438624(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e7f2438624(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e7f2438624(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (5888, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×16(12×16) {0..12}×{0..16} strided
        // m1 12×20(12×20) {0..12}×{0..20} strided
        // m2 16×20(16×20) {0..16}×{0..20} strided
        // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 12×20(12×20) {0..12}×{0..20} strided({0..12}×{0..20})[0, -1]×m2 16×20(16×20) {0..16}×{0..20} strided({0..16}×{0..20})[1, -1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[368 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[352];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 320 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[1, 0])
              tensorforge::intel_esimd::simd<int32_t, 16> v4_lead = tensorforge::intel_esimd::simd<int32_t, 16>(0, 1);
              #pragma unroll
              for (int32_t v5_i0 = 0; v5_i0 < 1; ++v5_i0) {
                int32_t v7_lead = v5_i0 * 16;
                #pragma unroll
                for (int32_t v6_i1 = 0; v6_i1 < 20; ++v6_i1) {
                  int32_t v9_a = v6_i1 * 16;
                  int32_t v10_a = v7_lead + v9_a;
                  tensorforge::intel_esimd::simd<float, 16> v15_data;
                  v15_data.copy_from(glb_m2 + ((v7_lead + v9_a)));
                  int32_t v19_a = v7_lead + (v6_i1 * 17);
                  s0[v19_a] = v15_data;
                }
              }
              // wait(s0 = load{g>s}(glb_m2[1, 0]));
              float r0[16]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir0[16]{};
              tensorforge::intel_esimd::simd_mask<16> v23_g = v4_lead < 12;
              int32_t v26_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v30_data(0.0f);
              v30_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[0_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v31_data(0.0f);
              v31_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[0]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v33_data(0.0f);
              v33_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v33_data + (v30_data * v31_data)).copy_to(ir0 + (0));
              }
              int32_t v37_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v42_data(0.0f);
              v42_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[1]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v44_data(0.0f);
              v44_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v44_data + (v30_data * v42_data)).copy_to(ir0 + (1));
              }
              int32_t v48_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v53_data(0.0f);
              v53_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[2]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v55_data(0.0f);
              v55_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v55_data + (v30_data * v53_data)).copy_to(ir0 + (2));
              }
              int32_t v59_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v64_data(0.0f);
              v64_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[3]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v66_data(0.0f);
              v66_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v66_data + (v30_data * v64_data)).copy_to(ir0 + (3));
              }
              int32_t v70_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v75_data(0.0f);
              v75_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[4]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v77_data(0.0f);
              v77_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v77_data + (v30_data * v75_data)).copy_to(ir0 + (4));
              }
              int32_t v81_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v86_data(0.0f);
              v86_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[5]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v88_data(0.0f);
              v88_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v88_data + (v30_data * v86_data)).copy_to(ir0 + (5));
              }
              int32_t v92_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v97_data(0.0f);
              v97_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[6]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v99_data(0.0f);
              v99_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v99_data + (v30_data * v97_data)).copy_to(ir0 + (6));
              }
              int32_t v103_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v108_data(0.0f);
              v108_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[7]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v110_data(0.0f);
              v110_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v110_data + (v30_data * v108_data)).copy_to(ir0 + (7));
              }
              int32_t v114_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v119_data(0.0f);
              v119_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[8]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v121_data(0.0f);
              v121_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v121_data + (v30_data * v119_data)).copy_to(ir0 + (8));
              }
              int32_t v125_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v130_data(0.0f);
              v130_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[9]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v132_data(0.0f);
              v132_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v132_data + (v30_data * v130_data)).copy_to(ir0 + (9));
              }
              int32_t v136_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v141_data(0.0f);
              v141_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[10]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v143_data(0.0f);
              v143_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v143_data + (v30_data * v141_data)).copy_to(ir0 + (10));
              }
              int32_t v147_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v152_data(0.0f);
              v152_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[11]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v154_data(0.0f);
              v154_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v154_data + (v30_data * v152_data)).copy_to(ir0 + (11));
              }
              int32_t v158_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v163_data(0.0f);
              v163_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[12]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v165_data(0.0f);
              v165_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v165_data + (v30_data * v163_data)).copy_to(ir0 + (12));
              }
              int32_t v169_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v174_data(0.0f);
              v174_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[13]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v176_data(0.0f);
              v176_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v176_data + (v30_data * v174_data)).copy_to(ir0 + (13));
              }
              int32_t v180_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v185_data(0.0f);
              v185_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[14]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v187_data(0.0f);
              v187_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v187_data + (v30_data * v185_data)).copy_to(ir0 + (14));
              }
              int32_t v191_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v196_data(0.0f);
              v196_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[15]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v198_data(0.0f);
              v198_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v198_data + (v30_data * v196_data)).copy_to(ir0 + (15));
              }
              int32_t v204_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v208_data(0.0f);
              v208_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[12_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v209_data(0.0f);
              v209_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[17]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v211_data(0.0f);
              v211_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v211_data + (v208_data * v209_data)).copy_to(ir0 + (0));
              }
              int32_t v215_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v220_data(0.0f);
              v220_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[18]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v222_data(0.0f);
              v222_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v222_data + (v208_data * v220_data)).copy_to(ir0 + (1));
              }
              int32_t v226_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v231_data(0.0f);
              v231_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[19]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v233_data(0.0f);
              v233_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v233_data + (v208_data * v231_data)).copy_to(ir0 + (2));
              }
              int32_t v237_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v242_data(0.0f);
              v242_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[20]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v244_data(0.0f);
              v244_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v244_data + (v208_data * v242_data)).copy_to(ir0 + (3));
              }
              int32_t v248_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v253_data(0.0f);
              v253_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[21]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v255_data(0.0f);
              v255_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v255_data + (v208_data * v253_data)).copy_to(ir0 + (4));
              }
              int32_t v259_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v264_data(0.0f);
              v264_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[22]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v266_data(0.0f);
              v266_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v266_data + (v208_data * v264_data)).copy_to(ir0 + (5));
              }
              int32_t v270_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v275_data(0.0f);
              v275_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[23]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v277_data(0.0f);
              v277_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v277_data + (v208_data * v275_data)).copy_to(ir0 + (6));
              }
              int32_t v281_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v286_data(0.0f);
              v286_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[24]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v288_data(0.0f);
              v288_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v288_data + (v208_data * v286_data)).copy_to(ir0 + (7));
              }
              int32_t v292_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v297_data(0.0f);
              v297_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[25]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v299_data(0.0f);
              v299_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v299_data + (v208_data * v297_data)).copy_to(ir0 + (8));
              }
              int32_t v303_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v308_data(0.0f);
              v308_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[26]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v310_data(0.0f);
              v310_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v310_data + (v208_data * v308_data)).copy_to(ir0 + (9));
              }
              int32_t v314_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v319_data(0.0f);
              v319_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[27]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v321_data(0.0f);
              v321_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v321_data + (v208_data * v319_data)).copy_to(ir0 + (10));
              }
              int32_t v325_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v330_data(0.0f);
              v330_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[28]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v332_data(0.0f);
              v332_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v332_data + (v208_data * v330_data)).copy_to(ir0 + (11));
              }
              int32_t v336_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v341_data(0.0f);
              v341_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[29]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v343_data(0.0f);
              v343_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v343_data + (v208_data * v341_data)).copy_to(ir0 + (12));
              }
              int32_t v347_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v352_data(0.0f);
              v352_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[30]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v354_data(0.0f);
              v354_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v354_data + (v208_data * v352_data)).copy_to(ir0 + (13));
              }
              int32_t v358_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v363_data(0.0f);
              v363_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[31]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v365_data(0.0f);
              v365_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v365_data + (v208_data * v363_data)).copy_to(ir0 + (14));
              }
              int32_t v369_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v374_data(0.0f);
              v374_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v376_data(0.0f);
              v376_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v376_data + (v208_data * v374_data)).copy_to(ir0 + (15));
              }
              int32_t v382_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v386_data(0.0f);
              v386_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[24_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v387_data(0.0f);
              v387_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[34]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v389_data(0.0f);
              v389_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v389_data + (v386_data * v387_data)).copy_to(ir0 + (0));
              }
              int32_t v393_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v398_data(0.0f);
              v398_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[35]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v400_data(0.0f);
              v400_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v400_data + (v386_data * v398_data)).copy_to(ir0 + (1));
              }
              int32_t v404_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v409_data(0.0f);
              v409_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[36]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v411_data(0.0f);
              v411_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v411_data + (v386_data * v409_data)).copy_to(ir0 + (2));
              }
              int32_t v415_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v420_data(0.0f);
              v420_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[37]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v422_data(0.0f);
              v422_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v422_data + (v386_data * v420_data)).copy_to(ir0 + (3));
              }
              int32_t v426_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v431_data(0.0f);
              v431_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[38]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v433_data(0.0f);
              v433_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v433_data + (v386_data * v431_data)).copy_to(ir0 + (4));
              }
              int32_t v437_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v442_data(0.0f);
              v442_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[39]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v444_data(0.0f);
              v444_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v444_data + (v386_data * v442_data)).copy_to(ir0 + (5));
              }
              int32_t v448_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v453_data(0.0f);
              v453_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[40]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v455_data(0.0f);
              v455_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v455_data + (v386_data * v453_data)).copy_to(ir0 + (6));
              }
              int32_t v459_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v464_data(0.0f);
              v464_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[41]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v466_data(0.0f);
              v466_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v466_data + (v386_data * v464_data)).copy_to(ir0 + (7));
              }
              int32_t v470_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v475_data(0.0f);
              v475_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[42]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v477_data(0.0f);
              v477_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v477_data + (v386_data * v475_data)).copy_to(ir0 + (8));
              }
              int32_t v481_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v486_data(0.0f);
              v486_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[43]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v488_data(0.0f);
              v488_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v488_data + (v386_data * v486_data)).copy_to(ir0 + (9));
              }
              int32_t v492_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v497_data(0.0f);
              v497_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[44]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v499_data(0.0f);
              v499_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v499_data + (v386_data * v497_data)).copy_to(ir0 + (10));
              }
              int32_t v503_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v508_data(0.0f);
              v508_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[45]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v510_data(0.0f);
              v510_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v510_data + (v386_data * v508_data)).copy_to(ir0 + (11));
              }
              int32_t v514_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v519_data(0.0f);
              v519_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[46]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v521_data(0.0f);
              v521_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v521_data + (v386_data * v519_data)).copy_to(ir0 + (12));
              }
              int32_t v525_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v530_data(0.0f);
              v530_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[47]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v532_data(0.0f);
              v532_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v532_data + (v386_data * v530_data)).copy_to(ir0 + (13));
              }
              int32_t v536_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v541_data(0.0f);
              v541_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[48]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v543_data(0.0f);
              v543_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v543_data + (v386_data * v541_data)).copy_to(ir0 + (14));
              }
              int32_t v547_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v552_data(0.0f);
              v552_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[49]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v554_data(0.0f);
              v554_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v554_data + (v386_data * v552_data)).copy_to(ir0 + (15));
              }
              int32_t v560_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v564_data(0.0f);
              v564_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[36_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v565_data(0.0f);
              v565_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[51]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v567_data(0.0f);
              v567_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v567_data + (v564_data * v565_data)).copy_to(ir0 + (0));
              }
              int32_t v571_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v576_data(0.0f);
              v576_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[52]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v578_data(0.0f);
              v578_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v578_data + (v564_data * v576_data)).copy_to(ir0 + (1));
              }
              int32_t v582_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v587_data(0.0f);
              v587_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[53]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v589_data(0.0f);
              v589_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v589_data + (v564_data * v587_data)).copy_to(ir0 + (2));
              }
              int32_t v593_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v598_data(0.0f);
              v598_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[54]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v600_data(0.0f);
              v600_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v600_data + (v564_data * v598_data)).copy_to(ir0 + (3));
              }
              int32_t v604_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v609_data(0.0f);
              v609_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[55]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v611_data(0.0f);
              v611_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v611_data + (v564_data * v609_data)).copy_to(ir0 + (4));
              }
              int32_t v615_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v620_data(0.0f);
              v620_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[56]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v622_data(0.0f);
              v622_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v622_data + (v564_data * v620_data)).copy_to(ir0 + (5));
              }
              int32_t v626_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v631_data(0.0f);
              v631_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[57]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v633_data(0.0f);
              v633_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v633_data + (v564_data * v631_data)).copy_to(ir0 + (6));
              }
              int32_t v637_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v642_data(0.0f);
              v642_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[58]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v644_data(0.0f);
              v644_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v644_data + (v564_data * v642_data)).copy_to(ir0 + (7));
              }
              int32_t v648_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v653_data(0.0f);
              v653_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[59]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v655_data(0.0f);
              v655_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v655_data + (v564_data * v653_data)).copy_to(ir0 + (8));
              }
              int32_t v659_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v664_data(0.0f);
              v664_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[60]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v666_data(0.0f);
              v666_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v666_data + (v564_data * v664_data)).copy_to(ir0 + (9));
              }
              int32_t v670_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v675_data(0.0f);
              v675_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[61]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v677_data(0.0f);
              v677_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v677_data + (v564_data * v675_data)).copy_to(ir0 + (10));
              }
              int32_t v681_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v686_data(0.0f);
              v686_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[62]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v688_data(0.0f);
              v688_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v688_data + (v564_data * v686_data)).copy_to(ir0 + (11));
              }
              int32_t v692_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v697_data(0.0f);
              v697_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[63]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v699_data(0.0f);
              v699_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v699_data + (v564_data * v697_data)).copy_to(ir0 + (12));
              }
              int32_t v703_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v708_data(0.0f);
              v708_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[64]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v710_data(0.0f);
              v710_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v710_data + (v564_data * v708_data)).copy_to(ir0 + (13));
              }
              int32_t v714_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v719_data(0.0f);
              v719_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[65]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v721_data(0.0f);
              v721_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v721_data + (v564_data * v719_data)).copy_to(ir0 + (14));
              }
              int32_t v725_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v730_data(0.0f);
              v730_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[66]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v732_data(0.0f);
              v732_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v732_data + (v564_data * v730_data)).copy_to(ir0 + (15));
              }
              int32_t v738_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v742_data(0.0f);
              v742_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[48_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v743_data(0.0f);
              v743_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[68]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v745_data(0.0f);
              v745_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v745_data + (v742_data * v743_data)).copy_to(ir0 + (0));
              }
              int32_t v749_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v754_data(0.0f);
              v754_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[69]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v756_data(0.0f);
              v756_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v756_data + (v742_data * v754_data)).copy_to(ir0 + (1));
              }
              int32_t v760_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v765_data(0.0f);
              v765_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[70]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v767_data(0.0f);
              v767_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v767_data + (v742_data * v765_data)).copy_to(ir0 + (2));
              }
              int32_t v771_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v776_data(0.0f);
              v776_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[71]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v778_data(0.0f);
              v778_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v778_data + (v742_data * v776_data)).copy_to(ir0 + (3));
              }
              int32_t v782_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v787_data(0.0f);
              v787_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[72]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v789_data(0.0f);
              v789_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v789_data + (v742_data * v787_data)).copy_to(ir0 + (4));
              }
              int32_t v793_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v798_data(0.0f);
              v798_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[73]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v800_data(0.0f);
              v800_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v800_data + (v742_data * v798_data)).copy_to(ir0 + (5));
              }
              int32_t v804_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v809_data(0.0f);
              v809_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[74]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v811_data(0.0f);
              v811_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v811_data + (v742_data * v809_data)).copy_to(ir0 + (6));
              }
              int32_t v815_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v820_data(0.0f);
              v820_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[75]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v822_data(0.0f);
              v822_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v822_data + (v742_data * v820_data)).copy_to(ir0 + (7));
              }
              int32_t v826_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v831_data(0.0f);
              v831_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[76]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v833_data(0.0f);
              v833_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v833_data + (v742_data * v831_data)).copy_to(ir0 + (8));
              }
              int32_t v837_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v842_data(0.0f);
              v842_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[77]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v844_data(0.0f);
              v844_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v844_data + (v742_data * v842_data)).copy_to(ir0 + (9));
              }
              int32_t v848_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v853_data(0.0f);
              v853_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[78]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v855_data(0.0f);
              v855_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v855_data + (v742_data * v853_data)).copy_to(ir0 + (10));
              }
              int32_t v859_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v864_data(0.0f);
              v864_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[79]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v866_data(0.0f);
              v866_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v866_data + (v742_data * v864_data)).copy_to(ir0 + (11));
              }
              int32_t v870_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v875_data(0.0f);
              v875_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[80]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v877_data(0.0f);
              v877_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v877_data + (v742_data * v875_data)).copy_to(ir0 + (12));
              }
              int32_t v881_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v886_data(0.0f);
              v886_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[81]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v888_data(0.0f);
              v888_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v888_data + (v742_data * v886_data)).copy_to(ir0 + (13));
              }
              int32_t v892_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v897_data(0.0f);
              v897_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[82]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v899_data(0.0f);
              v899_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v899_data + (v742_data * v897_data)).copy_to(ir0 + (14));
              }
              int32_t v903_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v908_data(0.0f);
              v908_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[83]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v910_data(0.0f);
              v910_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v910_data + (v742_data * v908_data)).copy_to(ir0 + (15));
              }
              int32_t v916_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v920_data(0.0f);
              v920_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[60_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v921_data(0.0f);
              v921_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[85]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v923_data(0.0f);
              v923_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v923_data + (v920_data * v921_data)).copy_to(ir0 + (0));
              }
              int32_t v927_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v932_data(0.0f);
              v932_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[86]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v934_data(0.0f);
              v934_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v934_data + (v920_data * v932_data)).copy_to(ir0 + (1));
              }
              int32_t v938_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v943_data(0.0f);
              v943_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[87]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v945_data(0.0f);
              v945_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v945_data + (v920_data * v943_data)).copy_to(ir0 + (2));
              }
              int32_t v949_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v954_data(0.0f);
              v954_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[88]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v956_data(0.0f);
              v956_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v956_data + (v920_data * v954_data)).copy_to(ir0 + (3));
              }
              int32_t v960_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v965_data(0.0f);
              v965_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[89]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v967_data(0.0f);
              v967_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v967_data + (v920_data * v965_data)).copy_to(ir0 + (4));
              }
              int32_t v971_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v976_data(0.0f);
              v976_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[90]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v978_data(0.0f);
              v978_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v978_data + (v920_data * v976_data)).copy_to(ir0 + (5));
              }
              int32_t v982_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v987_data(0.0f);
              v987_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[91]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v989_data(0.0f);
              v989_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v989_data + (v920_data * v987_data)).copy_to(ir0 + (6));
              }
              int32_t v993_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v998_data(0.0f);
              v998_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[92]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1000_data(0.0f);
              v1000_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v1000_data + (v920_data * v998_data)).copy_to(ir0 + (7));
              }
              int32_t v1004_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1009_data(0.0f);
              v1009_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[93]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1011_data(0.0f);
              v1011_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v1011_data + (v920_data * v1009_data)).copy_to(ir0 + (8));
              }
              int32_t v1015_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1020_data(0.0f);
              v1020_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[94]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1022_data(0.0f);
              v1022_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v1022_data + (v920_data * v1020_data)).copy_to(ir0 + (9));
              }
              int32_t v1026_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1031_data(0.0f);
              v1031_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[95]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1033_data(0.0f);
              v1033_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v1033_data + (v920_data * v1031_data)).copy_to(ir0 + (10));
              }
              int32_t v1037_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1042_data(0.0f);
              v1042_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[96]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1044_data(0.0f);
              v1044_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v1044_data + (v920_data * v1042_data)).copy_to(ir0 + (11));
              }
              int32_t v1048_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1053_data(0.0f);
              v1053_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[97]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1055_data(0.0f);
              v1055_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v1055_data + (v920_data * v1053_data)).copy_to(ir0 + (12));
              }
              int32_t v1059_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1064_data(0.0f);
              v1064_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[98]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1066_data(0.0f);
              v1066_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v1066_data + (v920_data * v1064_data)).copy_to(ir0 + (13));
              }
              int32_t v1070_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1075_data(0.0f);
              v1075_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[99]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1077_data(0.0f);
              v1077_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v1077_data + (v920_data * v1075_data)).copy_to(ir0 + (14));
              }
              int32_t v1081_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1086_data(0.0f);
              v1086_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[100]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1088_data(0.0f);
              v1088_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v1088_data + (v920_data * v1086_data)).copy_to(ir0 + (15));
              }
              int32_t v1094_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1098_data(0.0f);
              v1098_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[72_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1099_data(0.0f);
              v1099_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[102]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1101_data(0.0f);
              v1101_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v1101_data + (v1098_data * v1099_data)).copy_to(ir0 + (0));
              }
              int32_t v1105_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1110_data(0.0f);
              v1110_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[103]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1112_data(0.0f);
              v1112_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v1112_data + (v1098_data * v1110_data)).copy_to(ir0 + (1));
              }
              int32_t v1116_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1121_data(0.0f);
              v1121_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[104]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1123_data(0.0f);
              v1123_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v1123_data + (v1098_data * v1121_data)).copy_to(ir0 + (2));
              }
              int32_t v1127_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1132_data(0.0f);
              v1132_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[105]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1134_data(0.0f);
              v1134_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v1134_data + (v1098_data * v1132_data)).copy_to(ir0 + (3));
              }
              int32_t v1138_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1143_data(0.0f);
              v1143_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[106]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1145_data(0.0f);
              v1145_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v1145_data + (v1098_data * v1143_data)).copy_to(ir0 + (4));
              }
              int32_t v1149_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1154_data(0.0f);
              v1154_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[107]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1156_data(0.0f);
              v1156_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v1156_data + (v1098_data * v1154_data)).copy_to(ir0 + (5));
              }
              int32_t v1160_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1165_data(0.0f);
              v1165_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[108]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1167_data(0.0f);
              v1167_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v1167_data + (v1098_data * v1165_data)).copy_to(ir0 + (6));
              }
              int32_t v1171_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1176_data(0.0f);
              v1176_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[109]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1178_data(0.0f);
              v1178_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v1178_data + (v1098_data * v1176_data)).copy_to(ir0 + (7));
              }
              int32_t v1182_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1187_data(0.0f);
              v1187_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[110]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1189_data(0.0f);
              v1189_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v1189_data + (v1098_data * v1187_data)).copy_to(ir0 + (8));
              }
              int32_t v1193_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1198_data(0.0f);
              v1198_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[111]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1200_data(0.0f);
              v1200_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v1200_data + (v1098_data * v1198_data)).copy_to(ir0 + (9));
              }
              int32_t v1204_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1209_data(0.0f);
              v1209_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[112]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1211_data(0.0f);
              v1211_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v1211_data + (v1098_data * v1209_data)).copy_to(ir0 + (10));
              }
              int32_t v1215_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1220_data(0.0f);
              v1220_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[113]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1222_data(0.0f);
              v1222_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v1222_data + (v1098_data * v1220_data)).copy_to(ir0 + (11));
              }
              int32_t v1226_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1231_data(0.0f);
              v1231_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[114]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1233_data(0.0f);
              v1233_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v1233_data + (v1098_data * v1231_data)).copy_to(ir0 + (12));
              }
              int32_t v1237_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1242_data(0.0f);
              v1242_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[115]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1244_data(0.0f);
              v1244_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v1244_data + (v1098_data * v1242_data)).copy_to(ir0 + (13));
              }
              int32_t v1248_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1253_data(0.0f);
              v1253_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[116]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1255_data(0.0f);
              v1255_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v1255_data + (v1098_data * v1253_data)).copy_to(ir0 + (14));
              }
              int32_t v1259_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1264_data(0.0f);
              v1264_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[117]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1266_data(0.0f);
              v1266_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v1266_data + (v1098_data * v1264_data)).copy_to(ir0 + (15));
              }
              int32_t v1272_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1276_data(0.0f);
              v1276_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[84_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1277_data(0.0f);
              v1277_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[119]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1279_data(0.0f);
              v1279_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v1279_data + (v1276_data * v1277_data)).copy_to(ir0 + (0));
              }
              int32_t v1283_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1288_data(0.0f);
              v1288_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[120]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1290_data(0.0f);
              v1290_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v1290_data + (v1276_data * v1288_data)).copy_to(ir0 + (1));
              }
              int32_t v1294_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1299_data(0.0f);
              v1299_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[121]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1301_data(0.0f);
              v1301_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v1301_data + (v1276_data * v1299_data)).copy_to(ir0 + (2));
              }
              int32_t v1305_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1310_data(0.0f);
              v1310_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[122]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1312_data(0.0f);
              v1312_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v1312_data + (v1276_data * v1310_data)).copy_to(ir0 + (3));
              }
              int32_t v1316_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1321_data(0.0f);
              v1321_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[123]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1323_data(0.0f);
              v1323_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v1323_data + (v1276_data * v1321_data)).copy_to(ir0 + (4));
              }
              int32_t v1327_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1332_data(0.0f);
              v1332_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[124]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1334_data(0.0f);
              v1334_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v1334_data + (v1276_data * v1332_data)).copy_to(ir0 + (5));
              }
              int32_t v1338_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1343_data(0.0f);
              v1343_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[125]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1345_data(0.0f);
              v1345_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v1345_data + (v1276_data * v1343_data)).copy_to(ir0 + (6));
              }
              int32_t v1349_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1354_data(0.0f);
              v1354_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[126]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1356_data(0.0f);
              v1356_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v1356_data + (v1276_data * v1354_data)).copy_to(ir0 + (7));
              }
              int32_t v1360_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1365_data(0.0f);
              v1365_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[127]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1367_data(0.0f);
              v1367_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v1367_data + (v1276_data * v1365_data)).copy_to(ir0 + (8));
              }
              int32_t v1371_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1376_data(0.0f);
              v1376_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[128]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1378_data(0.0f);
              v1378_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v1378_data + (v1276_data * v1376_data)).copy_to(ir0 + (9));
              }
              int32_t v1382_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1387_data(0.0f);
              v1387_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[129]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1389_data(0.0f);
              v1389_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v1389_data + (v1276_data * v1387_data)).copy_to(ir0 + (10));
              }
              int32_t v1393_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1398_data(0.0f);
              v1398_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[130]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1400_data(0.0f);
              v1400_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v1400_data + (v1276_data * v1398_data)).copy_to(ir0 + (11));
              }
              int32_t v1404_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1409_data(0.0f);
              v1409_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[131]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1411_data(0.0f);
              v1411_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v1411_data + (v1276_data * v1409_data)).copy_to(ir0 + (12));
              }
              int32_t v1415_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1420_data(0.0f);
              v1420_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[132]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1422_data(0.0f);
              v1422_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v1422_data + (v1276_data * v1420_data)).copy_to(ir0 + (13));
              }
              int32_t v1426_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1431_data(0.0f);
              v1431_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[133]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1433_data(0.0f);
              v1433_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v1433_data + (v1276_data * v1431_data)).copy_to(ir0 + (14));
              }
              int32_t v1437_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1442_data(0.0f);
              v1442_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[134]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1444_data(0.0f);
              v1444_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v1444_data + (v1276_data * v1442_data)).copy_to(ir0 + (15));
              }
              int32_t v1450_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1454_data(0.0f);
              v1454_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[96_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1455_data(0.0f);
              v1455_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[136]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1457_data(0.0f);
              v1457_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v1457_data + (v1454_data * v1455_data)).copy_to(ir0 + (0));
              }
              int32_t v1461_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1466_data(0.0f);
              v1466_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[137]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1468_data(0.0f);
              v1468_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v1468_data + (v1454_data * v1466_data)).copy_to(ir0 + (1));
              }
              int32_t v1472_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1477_data(0.0f);
              v1477_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[138]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1479_data(0.0f);
              v1479_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v1479_data + (v1454_data * v1477_data)).copy_to(ir0 + (2));
              }
              int32_t v1483_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1488_data(0.0f);
              v1488_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[139]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1490_data(0.0f);
              v1490_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v1490_data + (v1454_data * v1488_data)).copy_to(ir0 + (3));
              }
              int32_t v1494_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1499_data(0.0f);
              v1499_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[140]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1501_data(0.0f);
              v1501_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v1501_data + (v1454_data * v1499_data)).copy_to(ir0 + (4));
              }
              int32_t v1505_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1510_data(0.0f);
              v1510_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[141]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1512_data(0.0f);
              v1512_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v1512_data + (v1454_data * v1510_data)).copy_to(ir0 + (5));
              }
              int32_t v1516_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1521_data(0.0f);
              v1521_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[142]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1523_data(0.0f);
              v1523_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v1523_data + (v1454_data * v1521_data)).copy_to(ir0 + (6));
              }
              int32_t v1527_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1532_data(0.0f);
              v1532_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[143]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1534_data(0.0f);
              v1534_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v1534_data + (v1454_data * v1532_data)).copy_to(ir0 + (7));
              }
              int32_t v1538_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1543_data(0.0f);
              v1543_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[144]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1545_data(0.0f);
              v1545_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v1545_data + (v1454_data * v1543_data)).copy_to(ir0 + (8));
              }
              int32_t v1549_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1554_data(0.0f);
              v1554_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[145]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1556_data(0.0f);
              v1556_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v1556_data + (v1454_data * v1554_data)).copy_to(ir0 + (9));
              }
              int32_t v1560_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1565_data(0.0f);
              v1565_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[146]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1567_data(0.0f);
              v1567_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v1567_data + (v1454_data * v1565_data)).copy_to(ir0 + (10));
              }
              int32_t v1571_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1576_data(0.0f);
              v1576_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[147]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1578_data(0.0f);
              v1578_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v1578_data + (v1454_data * v1576_data)).copy_to(ir0 + (11));
              }
              int32_t v1582_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1587_data(0.0f);
              v1587_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[148]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1589_data(0.0f);
              v1589_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v1589_data + (v1454_data * v1587_data)).copy_to(ir0 + (12));
              }
              int32_t v1593_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1598_data(0.0f);
              v1598_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[149]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1600_data(0.0f);
              v1600_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v1600_data + (v1454_data * v1598_data)).copy_to(ir0 + (13));
              }
              int32_t v1604_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1609_data(0.0f);
              v1609_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[150]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1611_data(0.0f);
              v1611_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v1611_data + (v1454_data * v1609_data)).copy_to(ir0 + (14));
              }
              int32_t v1615_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1620_data(0.0f);
              v1620_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[151]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1622_data(0.0f);
              v1622_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v1622_data + (v1454_data * v1620_data)).copy_to(ir0 + (15));
              }
              int32_t v1628_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1632_data(0.0f);
              v1632_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[108_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1633_data(0.0f);
              v1633_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[153]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1635_data(0.0f);
              v1635_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v1635_data + (v1632_data * v1633_data)).copy_to(ir0 + (0));
              }
              int32_t v1639_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1644_data(0.0f);
              v1644_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[154]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1646_data(0.0f);
              v1646_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v1646_data + (v1632_data * v1644_data)).copy_to(ir0 + (1));
              }
              int32_t v1650_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1655_data(0.0f);
              v1655_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[155]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1657_data(0.0f);
              v1657_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v1657_data + (v1632_data * v1655_data)).copy_to(ir0 + (2));
              }
              int32_t v1661_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1666_data(0.0f);
              v1666_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[156]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1668_data(0.0f);
              v1668_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v1668_data + (v1632_data * v1666_data)).copy_to(ir0 + (3));
              }
              int32_t v1672_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1677_data(0.0f);
              v1677_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[157]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1679_data(0.0f);
              v1679_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v1679_data + (v1632_data * v1677_data)).copy_to(ir0 + (4));
              }
              int32_t v1683_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1688_data(0.0f);
              v1688_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[158]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1690_data(0.0f);
              v1690_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v1690_data + (v1632_data * v1688_data)).copy_to(ir0 + (5));
              }
              int32_t v1694_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1699_data(0.0f);
              v1699_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[159]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1701_data(0.0f);
              v1701_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v1701_data + (v1632_data * v1699_data)).copy_to(ir0 + (6));
              }
              int32_t v1705_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1710_data(0.0f);
              v1710_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[160]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1712_data(0.0f);
              v1712_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v1712_data + (v1632_data * v1710_data)).copy_to(ir0 + (7));
              }
              int32_t v1716_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1721_data(0.0f);
              v1721_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[161]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1723_data(0.0f);
              v1723_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v1723_data + (v1632_data * v1721_data)).copy_to(ir0 + (8));
              }
              int32_t v1727_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1732_data(0.0f);
              v1732_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[162]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1734_data(0.0f);
              v1734_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v1734_data + (v1632_data * v1732_data)).copy_to(ir0 + (9));
              }
              int32_t v1738_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1743_data(0.0f);
              v1743_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[163]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1745_data(0.0f);
              v1745_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v1745_data + (v1632_data * v1743_data)).copy_to(ir0 + (10));
              }
              int32_t v1749_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1754_data(0.0f);
              v1754_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[164]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1756_data(0.0f);
              v1756_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v1756_data + (v1632_data * v1754_data)).copy_to(ir0 + (11));
              }
              int32_t v1760_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1765_data(0.0f);
              v1765_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[165]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1767_data(0.0f);
              v1767_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v1767_data + (v1632_data * v1765_data)).copy_to(ir0 + (12));
              }
              int32_t v1771_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1776_data(0.0f);
              v1776_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[166]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1778_data(0.0f);
              v1778_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v1778_data + (v1632_data * v1776_data)).copy_to(ir0 + (13));
              }
              int32_t v1782_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1787_data(0.0f);
              v1787_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[167]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1789_data(0.0f);
              v1789_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v1789_data + (v1632_data * v1787_data)).copy_to(ir0 + (14));
              }
              int32_t v1793_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1798_data(0.0f);
              v1798_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[168]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1800_data(0.0f);
              v1800_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v1800_data + (v1632_data * v1798_data)).copy_to(ir0 + (15));
              }
              int32_t v1806_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1810_data(0.0f);
              v1810_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[120_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1811_data(0.0f);
              v1811_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[170]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1813_data(0.0f);
              v1813_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v1813_data + (v1810_data * v1811_data)).copy_to(ir0 + (0));
              }
              int32_t v1817_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1822_data(0.0f);
              v1822_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[171]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1824_data(0.0f);
              v1824_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v1824_data + (v1810_data * v1822_data)).copy_to(ir0 + (1));
              }
              int32_t v1828_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1833_data(0.0f);
              v1833_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[172]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1835_data(0.0f);
              v1835_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v1835_data + (v1810_data * v1833_data)).copy_to(ir0 + (2));
              }
              int32_t v1839_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1844_data(0.0f);
              v1844_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[173]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1846_data(0.0f);
              v1846_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v1846_data + (v1810_data * v1844_data)).copy_to(ir0 + (3));
              }
              int32_t v1850_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1855_data(0.0f);
              v1855_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[174]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1857_data(0.0f);
              v1857_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v1857_data + (v1810_data * v1855_data)).copy_to(ir0 + (4));
              }
              int32_t v1861_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1866_data(0.0f);
              v1866_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[175]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1868_data(0.0f);
              v1868_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v1868_data + (v1810_data * v1866_data)).copy_to(ir0 + (5));
              }
              int32_t v1872_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1877_data(0.0f);
              v1877_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[176]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1879_data(0.0f);
              v1879_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v1879_data + (v1810_data * v1877_data)).copy_to(ir0 + (6));
              }
              int32_t v1883_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1888_data(0.0f);
              v1888_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[177]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1890_data(0.0f);
              v1890_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v1890_data + (v1810_data * v1888_data)).copy_to(ir0 + (7));
              }
              int32_t v1894_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1899_data(0.0f);
              v1899_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[178]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1901_data(0.0f);
              v1901_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v1901_data + (v1810_data * v1899_data)).copy_to(ir0 + (8));
              }
              int32_t v1905_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1910_data(0.0f);
              v1910_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[179]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1912_data(0.0f);
              v1912_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v1912_data + (v1810_data * v1910_data)).copy_to(ir0 + (9));
              }
              int32_t v1916_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1921_data(0.0f);
              v1921_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[180]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1923_data(0.0f);
              v1923_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v1923_data + (v1810_data * v1921_data)).copy_to(ir0 + (10));
              }
              int32_t v1927_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1932_data(0.0f);
              v1932_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[181]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1934_data(0.0f);
              v1934_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v1934_data + (v1810_data * v1932_data)).copy_to(ir0 + (11));
              }
              int32_t v1938_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1943_data(0.0f);
              v1943_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[182]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1945_data(0.0f);
              v1945_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v1945_data + (v1810_data * v1943_data)).copy_to(ir0 + (12));
              }
              int32_t v1949_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1954_data(0.0f);
              v1954_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[183]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1956_data(0.0f);
              v1956_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v1956_data + (v1810_data * v1954_data)).copy_to(ir0 + (13));
              }
              int32_t v1960_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1965_data(0.0f);
              v1965_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[184]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1967_data(0.0f);
              v1967_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v1967_data + (v1810_data * v1965_data)).copy_to(ir0 + (14));
              }
              int32_t v1971_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1976_data(0.0f);
              v1976_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[185]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1978_data(0.0f);
              v1978_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v1978_data + (v1810_data * v1976_data)).copy_to(ir0 + (15));
              }
              int32_t v1984_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1988_data(0.0f);
              v1988_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[132_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1989_data(0.0f);
              v1989_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[187]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v1991_data(0.0f);
              v1991_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v1991_data + (v1988_data * v1989_data)).copy_to(ir0 + (0));
              }
              int32_t v1995_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2000_data(0.0f);
              v2000_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[188]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2002_data(0.0f);
              v2002_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v2002_data + (v1988_data * v2000_data)).copy_to(ir0 + (1));
              }
              int32_t v2006_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2011_data(0.0f);
              v2011_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[189]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2013_data(0.0f);
              v2013_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v2013_data + (v1988_data * v2011_data)).copy_to(ir0 + (2));
              }
              int32_t v2017_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2022_data(0.0f);
              v2022_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[190]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2024_data(0.0f);
              v2024_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v2024_data + (v1988_data * v2022_data)).copy_to(ir0 + (3));
              }
              int32_t v2028_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2033_data(0.0f);
              v2033_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[191]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2035_data(0.0f);
              v2035_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v2035_data + (v1988_data * v2033_data)).copy_to(ir0 + (4));
              }
              int32_t v2039_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2044_data(0.0f);
              v2044_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[192]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2046_data(0.0f);
              v2046_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v2046_data + (v1988_data * v2044_data)).copy_to(ir0 + (5));
              }
              int32_t v2050_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2055_data(0.0f);
              v2055_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[193]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2057_data(0.0f);
              v2057_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v2057_data + (v1988_data * v2055_data)).copy_to(ir0 + (6));
              }
              int32_t v2061_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2066_data(0.0f);
              v2066_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[194]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2068_data(0.0f);
              v2068_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v2068_data + (v1988_data * v2066_data)).copy_to(ir0 + (7));
              }
              int32_t v2072_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2077_data(0.0f);
              v2077_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[195]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2079_data(0.0f);
              v2079_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v2079_data + (v1988_data * v2077_data)).copy_to(ir0 + (8));
              }
              int32_t v2083_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2088_data(0.0f);
              v2088_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[196]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2090_data(0.0f);
              v2090_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v2090_data + (v1988_data * v2088_data)).copy_to(ir0 + (9));
              }
              int32_t v2094_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2099_data(0.0f);
              v2099_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[197]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2101_data(0.0f);
              v2101_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v2101_data + (v1988_data * v2099_data)).copy_to(ir0 + (10));
              }
              int32_t v2105_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2110_data(0.0f);
              v2110_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[198]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2112_data(0.0f);
              v2112_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v2112_data + (v1988_data * v2110_data)).copy_to(ir0 + (11));
              }
              int32_t v2116_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2121_data(0.0f);
              v2121_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[199]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2123_data(0.0f);
              v2123_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v2123_data + (v1988_data * v2121_data)).copy_to(ir0 + (12));
              }
              int32_t v2127_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2132_data(0.0f);
              v2132_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[200]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2134_data(0.0f);
              v2134_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v2134_data + (v1988_data * v2132_data)).copy_to(ir0 + (13));
              }
              int32_t v2138_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2143_data(0.0f);
              v2143_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[201]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2145_data(0.0f);
              v2145_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v2145_data + (v1988_data * v2143_data)).copy_to(ir0 + (14));
              }
              int32_t v2149_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2154_data(0.0f);
              v2154_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[202]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2156_data(0.0f);
              v2156_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v2156_data + (v1988_data * v2154_data)).copy_to(ir0 + (15));
              }
              int32_t v2162_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2166_data(0.0f);
              v2166_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[144_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2167_data(0.0f);
              v2167_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[204]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2169_data(0.0f);
              v2169_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v2169_data + (v2166_data * v2167_data)).copy_to(ir0 + (0));
              }
              int32_t v2173_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2178_data(0.0f);
              v2178_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[205]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2180_data(0.0f);
              v2180_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v2180_data + (v2166_data * v2178_data)).copy_to(ir0 + (1));
              }
              int32_t v2184_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2189_data(0.0f);
              v2189_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[206]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2191_data(0.0f);
              v2191_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v2191_data + (v2166_data * v2189_data)).copy_to(ir0 + (2));
              }
              int32_t v2195_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2200_data(0.0f);
              v2200_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[207]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2202_data(0.0f);
              v2202_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v2202_data + (v2166_data * v2200_data)).copy_to(ir0 + (3));
              }
              int32_t v2206_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2211_data(0.0f);
              v2211_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[208]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2213_data(0.0f);
              v2213_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v2213_data + (v2166_data * v2211_data)).copy_to(ir0 + (4));
              }
              int32_t v2217_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2222_data(0.0f);
              v2222_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[209]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2224_data(0.0f);
              v2224_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v2224_data + (v2166_data * v2222_data)).copy_to(ir0 + (5));
              }
              int32_t v2228_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2233_data(0.0f);
              v2233_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[210]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2235_data(0.0f);
              v2235_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v2235_data + (v2166_data * v2233_data)).copy_to(ir0 + (6));
              }
              int32_t v2239_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2244_data(0.0f);
              v2244_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[211]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2246_data(0.0f);
              v2246_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v2246_data + (v2166_data * v2244_data)).copy_to(ir0 + (7));
              }
              int32_t v2250_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2255_data(0.0f);
              v2255_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[212]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2257_data(0.0f);
              v2257_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v2257_data + (v2166_data * v2255_data)).copy_to(ir0 + (8));
              }
              int32_t v2261_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2266_data(0.0f);
              v2266_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[213]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2268_data(0.0f);
              v2268_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v2268_data + (v2166_data * v2266_data)).copy_to(ir0 + (9));
              }
              int32_t v2272_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2277_data(0.0f);
              v2277_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[214]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2279_data(0.0f);
              v2279_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v2279_data + (v2166_data * v2277_data)).copy_to(ir0 + (10));
              }
              int32_t v2283_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2288_data(0.0f);
              v2288_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[215]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2290_data(0.0f);
              v2290_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v2290_data + (v2166_data * v2288_data)).copy_to(ir0 + (11));
              }
              int32_t v2294_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2299_data(0.0f);
              v2299_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[216]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2301_data(0.0f);
              v2301_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v2301_data + (v2166_data * v2299_data)).copy_to(ir0 + (12));
              }
              int32_t v2305_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2310_data(0.0f);
              v2310_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[217]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2312_data(0.0f);
              v2312_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v2312_data + (v2166_data * v2310_data)).copy_to(ir0 + (13));
              }
              int32_t v2316_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2321_data(0.0f);
              v2321_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[218]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2323_data(0.0f);
              v2323_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v2323_data + (v2166_data * v2321_data)).copy_to(ir0 + (14));
              }
              int32_t v2327_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v2332_data(0.0f);
              v2332_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[219]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2334_data(0.0f);
              v2334_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v2334_data + (v2166_data * v2332_data)).copy_to(ir0 + (15));
              }
              int32_t v2340_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2344_data(0.0f);
              v2344_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[156_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2345_data(0.0f);
              v2345_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[221]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2347_data(0.0f);
              v2347_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v2347_data + (v2344_data * v2345_data)).copy_to(ir0 + (0));
              }
              int32_t v2351_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2356_data(0.0f);
              v2356_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[222]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2358_data(0.0f);
              v2358_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v2358_data + (v2344_data * v2356_data)).copy_to(ir0 + (1));
              }
              int32_t v2362_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2367_data(0.0f);
              v2367_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[223]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2369_data(0.0f);
              v2369_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v2369_data + (v2344_data * v2367_data)).copy_to(ir0 + (2));
              }
              int32_t v2373_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2378_data(0.0f);
              v2378_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[224]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2380_data(0.0f);
              v2380_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v2380_data + (v2344_data * v2378_data)).copy_to(ir0 + (3));
              }
              int32_t v2384_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2389_data(0.0f);
              v2389_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[225]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2391_data(0.0f);
              v2391_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v2391_data + (v2344_data * v2389_data)).copy_to(ir0 + (4));
              }
              int32_t v2395_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2400_data(0.0f);
              v2400_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[226]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2402_data(0.0f);
              v2402_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v2402_data + (v2344_data * v2400_data)).copy_to(ir0 + (5));
              }
              int32_t v2406_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2411_data(0.0f);
              v2411_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[227]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2413_data(0.0f);
              v2413_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v2413_data + (v2344_data * v2411_data)).copy_to(ir0 + (6));
              }
              int32_t v2417_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2422_data(0.0f);
              v2422_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[228]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2424_data(0.0f);
              v2424_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v2424_data + (v2344_data * v2422_data)).copy_to(ir0 + (7));
              }
              int32_t v2428_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2433_data(0.0f);
              v2433_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[229]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2435_data(0.0f);
              v2435_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v2435_data + (v2344_data * v2433_data)).copy_to(ir0 + (8));
              }
              int32_t v2439_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2444_data(0.0f);
              v2444_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[230]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2446_data(0.0f);
              v2446_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v2446_data + (v2344_data * v2444_data)).copy_to(ir0 + (9));
              }
              int32_t v2450_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2455_data(0.0f);
              v2455_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[231]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2457_data(0.0f);
              v2457_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v2457_data + (v2344_data * v2455_data)).copy_to(ir0 + (10));
              }
              int32_t v2461_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2466_data(0.0f);
              v2466_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[232]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2468_data(0.0f);
              v2468_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v2468_data + (v2344_data * v2466_data)).copy_to(ir0 + (11));
              }
              int32_t v2472_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2477_data(0.0f);
              v2477_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[233]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2479_data(0.0f);
              v2479_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v2479_data + (v2344_data * v2477_data)).copy_to(ir0 + (12));
              }
              int32_t v2483_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2488_data(0.0f);
              v2488_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[234]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2490_data(0.0f);
              v2490_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v2490_data + (v2344_data * v2488_data)).copy_to(ir0 + (13));
              }
              int32_t v2494_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2499_data(0.0f);
              v2499_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[235]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2501_data(0.0f);
              v2501_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v2501_data + (v2344_data * v2499_data)).copy_to(ir0 + (14));
              }
              int32_t v2505_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v2510_data(0.0f);
              v2510_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[236]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2512_data(0.0f);
              v2512_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v2512_data + (v2344_data * v2510_data)).copy_to(ir0 + (15));
              }
              int32_t v2518_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2522_data(0.0f);
              v2522_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[168_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2523_data(0.0f);
              v2523_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[238]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2525_data(0.0f);
              v2525_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v2525_data + (v2522_data * v2523_data)).copy_to(ir0 + (0));
              }
              int32_t v2529_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2534_data(0.0f);
              v2534_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[239]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2536_data(0.0f);
              v2536_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v2536_data + (v2522_data * v2534_data)).copy_to(ir0 + (1));
              }
              int32_t v2540_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2545_data(0.0f);
              v2545_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[240]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2547_data(0.0f);
              v2547_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v2547_data + (v2522_data * v2545_data)).copy_to(ir0 + (2));
              }
              int32_t v2551_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2556_data(0.0f);
              v2556_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[241]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2558_data(0.0f);
              v2558_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v2558_data + (v2522_data * v2556_data)).copy_to(ir0 + (3));
              }
              int32_t v2562_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2567_data(0.0f);
              v2567_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[242]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2569_data(0.0f);
              v2569_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v2569_data + (v2522_data * v2567_data)).copy_to(ir0 + (4));
              }
              int32_t v2573_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2578_data(0.0f);
              v2578_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[243]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2580_data(0.0f);
              v2580_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v2580_data + (v2522_data * v2578_data)).copy_to(ir0 + (5));
              }
              int32_t v2584_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2589_data(0.0f);
              v2589_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[244]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2591_data(0.0f);
              v2591_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v2591_data + (v2522_data * v2589_data)).copy_to(ir0 + (6));
              }
              int32_t v2595_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2600_data(0.0f);
              v2600_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[245]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2602_data(0.0f);
              v2602_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v2602_data + (v2522_data * v2600_data)).copy_to(ir0 + (7));
              }
              int32_t v2606_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2611_data(0.0f);
              v2611_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[246]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2613_data(0.0f);
              v2613_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v2613_data + (v2522_data * v2611_data)).copy_to(ir0 + (8));
              }
              int32_t v2617_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2622_data(0.0f);
              v2622_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[247]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2624_data(0.0f);
              v2624_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v2624_data + (v2522_data * v2622_data)).copy_to(ir0 + (9));
              }
              int32_t v2628_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2633_data(0.0f);
              v2633_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[248]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2635_data(0.0f);
              v2635_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v2635_data + (v2522_data * v2633_data)).copy_to(ir0 + (10));
              }
              int32_t v2639_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2644_data(0.0f);
              v2644_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[249]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2646_data(0.0f);
              v2646_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v2646_data + (v2522_data * v2644_data)).copy_to(ir0 + (11));
              }
              int32_t v2650_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2655_data(0.0f);
              v2655_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[250]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2657_data(0.0f);
              v2657_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v2657_data + (v2522_data * v2655_data)).copy_to(ir0 + (12));
              }
              int32_t v2661_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2666_data(0.0f);
              v2666_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[251]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2668_data(0.0f);
              v2668_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v2668_data + (v2522_data * v2666_data)).copy_to(ir0 + (13));
              }
              int32_t v2672_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2677_data(0.0f);
              v2677_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[252]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2679_data(0.0f);
              v2679_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v2679_data + (v2522_data * v2677_data)).copy_to(ir0 + (14));
              }
              int32_t v2683_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v2688_data(0.0f);
              v2688_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[253]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2690_data(0.0f);
              v2690_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v2690_data + (v2522_data * v2688_data)).copy_to(ir0 + (15));
              }
              int32_t v2696_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2700_data(0.0f);
              v2700_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[180_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2701_data(0.0f);
              v2701_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[255]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2703_data(0.0f);
              v2703_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v2703_data + (v2700_data * v2701_data)).copy_to(ir0 + (0));
              }
              int32_t v2707_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2712_data(0.0f);
              v2712_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[256]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2714_data(0.0f);
              v2714_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v2714_data + (v2700_data * v2712_data)).copy_to(ir0 + (1));
              }
              int32_t v2718_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2723_data(0.0f);
              v2723_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[257]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2725_data(0.0f);
              v2725_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v2725_data + (v2700_data * v2723_data)).copy_to(ir0 + (2));
              }
              int32_t v2729_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2734_data(0.0f);
              v2734_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[258]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2736_data(0.0f);
              v2736_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v2736_data + (v2700_data * v2734_data)).copy_to(ir0 + (3));
              }
              int32_t v2740_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2745_data(0.0f);
              v2745_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[259]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2747_data(0.0f);
              v2747_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v2747_data + (v2700_data * v2745_data)).copy_to(ir0 + (4));
              }
              int32_t v2751_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2756_data(0.0f);
              v2756_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[260]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2758_data(0.0f);
              v2758_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v2758_data + (v2700_data * v2756_data)).copy_to(ir0 + (5));
              }
              int32_t v2762_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2767_data(0.0f);
              v2767_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[261]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2769_data(0.0f);
              v2769_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v2769_data + (v2700_data * v2767_data)).copy_to(ir0 + (6));
              }
              int32_t v2773_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2778_data(0.0f);
              v2778_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[262]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2780_data(0.0f);
              v2780_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v2780_data + (v2700_data * v2778_data)).copy_to(ir0 + (7));
              }
              int32_t v2784_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2789_data(0.0f);
              v2789_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[263]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2791_data(0.0f);
              v2791_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v2791_data + (v2700_data * v2789_data)).copy_to(ir0 + (8));
              }
              int32_t v2795_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2800_data(0.0f);
              v2800_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[264]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2802_data(0.0f);
              v2802_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v2802_data + (v2700_data * v2800_data)).copy_to(ir0 + (9));
              }
              int32_t v2806_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2811_data(0.0f);
              v2811_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[265]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2813_data(0.0f);
              v2813_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v2813_data + (v2700_data * v2811_data)).copy_to(ir0 + (10));
              }
              int32_t v2817_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2822_data(0.0f);
              v2822_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[266]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2824_data(0.0f);
              v2824_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v2824_data + (v2700_data * v2822_data)).copy_to(ir0 + (11));
              }
              int32_t v2828_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2833_data(0.0f);
              v2833_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[267]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2835_data(0.0f);
              v2835_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v2835_data + (v2700_data * v2833_data)).copy_to(ir0 + (12));
              }
              int32_t v2839_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2844_data(0.0f);
              v2844_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[268]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2846_data(0.0f);
              v2846_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v2846_data + (v2700_data * v2844_data)).copy_to(ir0 + (13));
              }
              int32_t v2850_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2855_data(0.0f);
              v2855_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[269]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2857_data(0.0f);
              v2857_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v2857_data + (v2700_data * v2855_data)).copy_to(ir0 + (14));
              }
              int32_t v2861_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v2866_data(0.0f);
              v2866_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[270]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2868_data(0.0f);
              v2868_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v2868_data + (v2700_data * v2866_data)).copy_to(ir0 + (15));
              }
              int32_t v2874_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v2878_data(0.0f);
              v2878_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[192_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2879_data(0.0f);
              v2879_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[272]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2881_data(0.0f);
              v2881_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v2881_data + (v2878_data * v2879_data)).copy_to(ir0 + (0));
              }
              int32_t v2885_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v2890_data(0.0f);
              v2890_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[273]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2892_data(0.0f);
              v2892_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v2892_data + (v2878_data * v2890_data)).copy_to(ir0 + (1));
              }
              int32_t v2896_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v2901_data(0.0f);
              v2901_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[274]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2903_data(0.0f);
              v2903_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v2903_data + (v2878_data * v2901_data)).copy_to(ir0 + (2));
              }
              int32_t v2907_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v2912_data(0.0f);
              v2912_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[275]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2914_data(0.0f);
              v2914_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v2914_data + (v2878_data * v2912_data)).copy_to(ir0 + (3));
              }
              int32_t v2918_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v2923_data(0.0f);
              v2923_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[276]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2925_data(0.0f);
              v2925_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v2925_data + (v2878_data * v2923_data)).copy_to(ir0 + (4));
              }
              int32_t v2929_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v2934_data(0.0f);
              v2934_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[277]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2936_data(0.0f);
              v2936_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v2936_data + (v2878_data * v2934_data)).copy_to(ir0 + (5));
              }
              int32_t v2940_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v2945_data(0.0f);
              v2945_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[278]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2947_data(0.0f);
              v2947_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v2947_data + (v2878_data * v2945_data)).copy_to(ir0 + (6));
              }
              int32_t v2951_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v2956_data(0.0f);
              v2956_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[279]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2958_data(0.0f);
              v2958_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v2958_data + (v2878_data * v2956_data)).copy_to(ir0 + (7));
              }
              int32_t v2962_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v2967_data(0.0f);
              v2967_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[280]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2969_data(0.0f);
              v2969_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v2969_data + (v2878_data * v2967_data)).copy_to(ir0 + (8));
              }
              int32_t v2973_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v2978_data(0.0f);
              v2978_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[281]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2980_data(0.0f);
              v2980_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v2980_data + (v2878_data * v2978_data)).copy_to(ir0 + (9));
              }
              int32_t v2984_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v2989_data(0.0f);
              v2989_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[282]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v2991_data(0.0f);
              v2991_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v2991_data + (v2878_data * v2989_data)).copy_to(ir0 + (10));
              }
              int32_t v2995_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v3000_data(0.0f);
              v3000_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[283]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3002_data(0.0f);
              v3002_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v3002_data + (v2878_data * v3000_data)).copy_to(ir0 + (11));
              }
              int32_t v3006_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v3011_data(0.0f);
              v3011_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[284]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3013_data(0.0f);
              v3013_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v3013_data + (v2878_data * v3011_data)).copy_to(ir0 + (12));
              }
              int32_t v3017_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v3022_data(0.0f);
              v3022_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[285]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3024_data(0.0f);
              v3024_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v3024_data + (v2878_data * v3022_data)).copy_to(ir0 + (13));
              }
              int32_t v3028_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v3033_data(0.0f);
              v3033_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[286]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3035_data(0.0f);
              v3035_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v3035_data + (v2878_data * v3033_data)).copy_to(ir0 + (14));
              }
              int32_t v3039_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v3044_data(0.0f);
              v3044_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[287]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3046_data(0.0f);
              v3046_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v3046_data + (v2878_data * v3044_data)).copy_to(ir0 + (15));
              }
              int32_t v3052_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3056_data(0.0f);
              v3056_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[204_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3057_data(0.0f);
              v3057_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[289]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3059_data(0.0f);
              v3059_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v3059_data + (v3056_data * v3057_data)).copy_to(ir0 + (0));
              }
              int32_t v3063_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3068_data(0.0f);
              v3068_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[290]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3070_data(0.0f);
              v3070_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v3070_data + (v3056_data * v3068_data)).copy_to(ir0 + (1));
              }
              int32_t v3074_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3079_data(0.0f);
              v3079_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[291]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3081_data(0.0f);
              v3081_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v3081_data + (v3056_data * v3079_data)).copy_to(ir0 + (2));
              }
              int32_t v3085_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3090_data(0.0f);
              v3090_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[292]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3092_data(0.0f);
              v3092_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v3092_data + (v3056_data * v3090_data)).copy_to(ir0 + (3));
              }
              int32_t v3096_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3101_data(0.0f);
              v3101_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[293]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3103_data(0.0f);
              v3103_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v3103_data + (v3056_data * v3101_data)).copy_to(ir0 + (4));
              }
              int32_t v3107_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3112_data(0.0f);
              v3112_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[294]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3114_data(0.0f);
              v3114_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v3114_data + (v3056_data * v3112_data)).copy_to(ir0 + (5));
              }
              int32_t v3118_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3123_data(0.0f);
              v3123_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[295]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3125_data(0.0f);
              v3125_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v3125_data + (v3056_data * v3123_data)).copy_to(ir0 + (6));
              }
              int32_t v3129_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3134_data(0.0f);
              v3134_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[296]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3136_data(0.0f);
              v3136_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v3136_data + (v3056_data * v3134_data)).copy_to(ir0 + (7));
              }
              int32_t v3140_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3145_data(0.0f);
              v3145_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[297]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3147_data(0.0f);
              v3147_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v3147_data + (v3056_data * v3145_data)).copy_to(ir0 + (8));
              }
              int32_t v3151_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3156_data(0.0f);
              v3156_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[298]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3158_data(0.0f);
              v3158_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v3158_data + (v3056_data * v3156_data)).copy_to(ir0 + (9));
              }
              int32_t v3162_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3167_data(0.0f);
              v3167_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[299]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3169_data(0.0f);
              v3169_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v3169_data + (v3056_data * v3167_data)).copy_to(ir0 + (10));
              }
              int32_t v3173_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3178_data(0.0f);
              v3178_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[300]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3180_data(0.0f);
              v3180_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v3180_data + (v3056_data * v3178_data)).copy_to(ir0 + (11));
              }
              int32_t v3184_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3189_data(0.0f);
              v3189_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[301]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3191_data(0.0f);
              v3191_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v3191_data + (v3056_data * v3189_data)).copy_to(ir0 + (12));
              }
              int32_t v3195_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3200_data(0.0f);
              v3200_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[302]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3202_data(0.0f);
              v3202_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v3202_data + (v3056_data * v3200_data)).copy_to(ir0 + (13));
              }
              int32_t v3206_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3211_data(0.0f);
              v3211_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[303]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3213_data(0.0f);
              v3213_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v3213_data + (v3056_data * v3211_data)).copy_to(ir0 + (14));
              }
              int32_t v3217_a = 0_i32 + 204;
              tensorforge::intel_esimd::simd<float, 16> v3222_data(0.0f);
              v3222_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[304]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3224_data(0.0f);
              v3224_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v3224_data + (v3056_data * v3222_data)).copy_to(ir0 + (15));
              }
              int32_t v3230_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3234_data(0.0f);
              v3234_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[216_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3235_data(0.0f);
              v3235_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[306]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3237_data(0.0f);
              v3237_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v3237_data + (v3234_data * v3235_data)).copy_to(ir0 + (0));
              }
              int32_t v3241_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3246_data(0.0f);
              v3246_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[307]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3248_data(0.0f);
              v3248_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v3248_data + (v3234_data * v3246_data)).copy_to(ir0 + (1));
              }
              int32_t v3252_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3257_data(0.0f);
              v3257_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[308]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3259_data(0.0f);
              v3259_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v3259_data + (v3234_data * v3257_data)).copy_to(ir0 + (2));
              }
              int32_t v3263_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3268_data(0.0f);
              v3268_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[309]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3270_data(0.0f);
              v3270_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v3270_data + (v3234_data * v3268_data)).copy_to(ir0 + (3));
              }
              int32_t v3274_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3279_data(0.0f);
              v3279_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[310]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3281_data(0.0f);
              v3281_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v3281_data + (v3234_data * v3279_data)).copy_to(ir0 + (4));
              }
              int32_t v3285_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3290_data(0.0f);
              v3290_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[311]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3292_data(0.0f);
              v3292_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v3292_data + (v3234_data * v3290_data)).copy_to(ir0 + (5));
              }
              int32_t v3296_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3301_data(0.0f);
              v3301_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[312]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3303_data(0.0f);
              v3303_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v3303_data + (v3234_data * v3301_data)).copy_to(ir0 + (6));
              }
              int32_t v3307_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3312_data(0.0f);
              v3312_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[313]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3314_data(0.0f);
              v3314_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v3314_data + (v3234_data * v3312_data)).copy_to(ir0 + (7));
              }
              int32_t v3318_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3323_data(0.0f);
              v3323_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[314]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3325_data(0.0f);
              v3325_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v3325_data + (v3234_data * v3323_data)).copy_to(ir0 + (8));
              }
              int32_t v3329_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3334_data(0.0f);
              v3334_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[315]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3336_data(0.0f);
              v3336_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v3336_data + (v3234_data * v3334_data)).copy_to(ir0 + (9));
              }
              int32_t v3340_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3345_data(0.0f);
              v3345_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[316]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3347_data(0.0f);
              v3347_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v3347_data + (v3234_data * v3345_data)).copy_to(ir0 + (10));
              }
              int32_t v3351_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3356_data(0.0f);
              v3356_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[317]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3358_data(0.0f);
              v3358_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v3358_data + (v3234_data * v3356_data)).copy_to(ir0 + (11));
              }
              int32_t v3362_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3367_data(0.0f);
              v3367_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[318]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3369_data(0.0f);
              v3369_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v3369_data + (v3234_data * v3367_data)).copy_to(ir0 + (12));
              }
              int32_t v3373_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3378_data(0.0f);
              v3378_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[319]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3380_data(0.0f);
              v3380_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v3380_data + (v3234_data * v3378_data)).copy_to(ir0 + (13));
              }
              int32_t v3384_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3389_data(0.0f);
              v3389_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[320]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3391_data(0.0f);
              v3391_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v3391_data + (v3234_data * v3389_data)).copy_to(ir0 + (14));
              }
              int32_t v3395_a = 0_i32 + 216;
              tensorforge::intel_esimd::simd<float, 16> v3400_data(0.0f);
              v3400_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[321]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3402_data(0.0f);
              v3402_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v3402_data + (v3234_data * v3400_data)).copy_to(ir0 + (15));
              }
              int32_t v3408_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3412_data(0.0f);
              v3412_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[228_i32]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3413_data(0.0f);
              v3413_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[323]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3415_data(0.0f);
              v3415_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v23_g);
              if (v23_g) {
                (v3415_data + (v3412_data * v3413_data)).copy_to(ir0 + (0));
              }
              int32_t v3419_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3424_data(0.0f);
              v3424_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[324]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3426_data(0.0f);
              v3426_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v23_g);
              if (v23_g) {
                (v3426_data + (v3412_data * v3424_data)).copy_to(ir0 + (1));
              }
              int32_t v3430_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3435_data(0.0f);
              v3435_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[325]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3437_data(0.0f);
              v3437_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v23_g);
              if (v23_g) {
                (v3437_data + (v3412_data * v3435_data)).copy_to(ir0 + (2));
              }
              int32_t v3441_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3446_data(0.0f);
              v3446_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[326]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3448_data(0.0f);
              v3448_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v23_g);
              if (v23_g) {
                (v3448_data + (v3412_data * v3446_data)).copy_to(ir0 + (3));
              }
              int32_t v3452_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3457_data(0.0f);
              v3457_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[327]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3459_data(0.0f);
              v3459_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v23_g);
              if (v23_g) {
                (v3459_data + (v3412_data * v3457_data)).copy_to(ir0 + (4));
              }
              int32_t v3463_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3468_data(0.0f);
              v3468_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[328]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3470_data(0.0f);
              v3470_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v23_g);
              if (v23_g) {
                (v3470_data + (v3412_data * v3468_data)).copy_to(ir0 + (5));
              }
              int32_t v3474_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3479_data(0.0f);
              v3479_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[329]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3481_data(0.0f);
              v3481_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v23_g);
              if (v23_g) {
                (v3481_data + (v3412_data * v3479_data)).copy_to(ir0 + (6));
              }
              int32_t v3485_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3490_data(0.0f);
              v3490_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[330]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3492_data(0.0f);
              v3492_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v23_g);
              if (v23_g) {
                (v3492_data + (v3412_data * v3490_data)).copy_to(ir0 + (7));
              }
              int32_t v3496_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3501_data(0.0f);
              v3501_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[331]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3503_data(0.0f);
              v3503_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v23_g);
              if (v23_g) {
                (v3503_data + (v3412_data * v3501_data)).copy_to(ir0 + (8));
              }
              int32_t v3507_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3512_data(0.0f);
              v3512_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[332]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3514_data(0.0f);
              v3514_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v23_g);
              if (v23_g) {
                (v3514_data + (v3412_data * v3512_data)).copy_to(ir0 + (9));
              }
              int32_t v3518_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3523_data(0.0f);
              v3523_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[333]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3525_data(0.0f);
              v3525_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v23_g);
              if (v23_g) {
                (v3525_data + (v3412_data * v3523_data)).copy_to(ir0 + (10));
              }
              int32_t v3529_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3534_data(0.0f);
              v3534_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[334]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3536_data(0.0f);
              v3536_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v23_g);
              if (v23_g) {
                (v3536_data + (v3412_data * v3534_data)).copy_to(ir0 + (11));
              }
              int32_t v3540_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3545_data(0.0f);
              v3545_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[335]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3547_data(0.0f);
              v3547_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v23_g);
              if (v23_g) {
                (v3547_data + (v3412_data * v3545_data)).copy_to(ir0 + (12));
              }
              int32_t v3551_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3556_data(0.0f);
              v3556_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[336]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3558_data(0.0f);
              v3558_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v23_g);
              if (v23_g) {
                (v3558_data + (v3412_data * v3556_data)).copy_to(ir0 + (13));
              }
              int32_t v3562_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3567_data(0.0f);
              v3567_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[337]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3569_data(0.0f);
              v3569_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v23_g);
              if (v23_g) {
                (v3569_data + (v3412_data * v3567_data)).copy_to(ir0 + (14));
              }
              int32_t v3573_a = 0_i32 + 228;
              tensorforge::intel_esimd::simd<float, 16> v3578_data(0.0f);
              v3578_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[338]), v23_g);
              tensorforge::intel_esimd::simd<float, 16> v3580_data(0.0f);
              v3580_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v23_g);
              if (v23_g) {
                (v3580_data + (v3412_data * v3578_data)).copy_to(ir0 + (15));
              }
              #pragma unroll
              for (int32_t v3584_n1 = 0; v3584_n1 < 16; ++v3584_n1) {
                int32_t v3585_a = 0 + v3584_n1;
                tensorforge::intel_esimd::simd<float, 16> v3587_data(0.0f);
                v3587_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[v3584_n1]), v23_g);
                if (v23_g) {
                  v3587_data.copy_to(r0 + (v3584_n1));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v3591_i1 = 0; v3591_i1 < 16; ++v3591_i1) {
                int32_t v3592_a = 0 + v3591_i1;
                tensorforge::intel_esimd::simd<float, 16> v3594_data(0.0f);
                v3594_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[v3591_i1]), v23_g);
                if (v23_g) {
                  v3594_data.copy_to(glb_m0 + ((v3591_i1 * 12)));
                }
              }
            }
          }
        }
      });
    }
  });
}

