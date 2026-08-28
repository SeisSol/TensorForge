// === base name ===
kernel_671a350836

// === header ===
void launcher_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_671a350836(const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_671a350836(stream, grid, block,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_671a350836(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, unsigned m0_extraOffset, const float* m1, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 64×13(64×13) {0..64}×{0..13} pointer_based
        // m1 6(6) {0..6} none
        // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based
        // t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..64}×{0..13}×{0..6})[0, 1, 2] = m0 64×13(64×13) {0..64}×{0..13} pointer_based({0..64}×{0..13})[0, 1]×m1 6(6) {0..6} none({0..6})[2]
        // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based({0..15}×{0..1}×{0..6})[0, 1, 2] += t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..15}×{0..1}×{0..6})[0, 1, 2]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const float *const __restrict__ glb_m1 = &m1[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float r0[156]{};
              // r0 = +(glb_m0 * glb_m1) + None
              // [(0, 64), (0, 13), (0, 6)] []
              tensorforge::intel_esimd::simd<int32_t, 32> v4_lead = tensorforge::intel_esimd::simd<int32_t, 32>(0, 1);
              int32_t v7_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v11_data;
              v11_data.copy_from(glb_m0 + (0_i32));
              float v12_data = glb_m1[0];
              tensorforge::intel_esimd::simd<float, 32> v14_data;
              v14_data.copy_from(r0 + (0));
              (v14_data + (v11_data * v12_data)).copy_to(r0 + (0));
              int32_t v18_a = 0_i32 + 0;
              float v23_data = glb_m1[1];
              tensorforge::intel_esimd::simd<float, 32> v25_data;
              v25_data.copy_from(r0 + (26));
              (v25_data + (v11_data * v23_data)).copy_to(r0 + (26));
              int32_t v29_a = 0_i32 + 0;
              float v34_data = glb_m1[2];
              tensorforge::intel_esimd::simd<float, 32> v36_data;
              v36_data.copy_from(r0 + (52));
              (v36_data + (v11_data * v34_data)).copy_to(r0 + (52));
              int32_t v40_a = 0_i32 + 0;
              float v45_data = glb_m1[3];
              tensorforge::intel_esimd::simd<float, 32> v47_data;
              v47_data.copy_from(r0 + (78));
              (v47_data + (v11_data * v45_data)).copy_to(r0 + (78));
              int32_t v51_a = 0_i32 + 0;
              float v56_data = glb_m1[4];
              tensorforge::intel_esimd::simd<float, 32> v58_data;
              v58_data.copy_from(r0 + (104));
              (v58_data + (v11_data * v56_data)).copy_to(r0 + (104));
              int32_t v62_a = 0_i32 + 0;
              float v67_data = glb_m1[5];
              tensorforge::intel_esimd::simd<float, 32> v69_data;
              v69_data.copy_from(r0 + (130));
              (v69_data + (v11_data * v67_data)).copy_to(r0 + (130));
              int32_t v73_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v77_data;
              v77_data.copy_from(glb_m0 + (64_i32));
              tensorforge::intel_esimd::simd<float, 32> v80_data;
              v80_data.copy_from(r0 + (2));
              (v80_data + (v77_data * v12_data)).copy_to(r0 + (2));
              int32_t v84_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v91_data;
              v91_data.copy_from(r0 + (28));
              (v91_data + (v77_data * v23_data)).copy_to(r0 + (28));
              int32_t v95_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v102_data;
              v102_data.copy_from(r0 + (54));
              (v102_data + (v77_data * v34_data)).copy_to(r0 + (54));
              int32_t v106_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v113_data;
              v113_data.copy_from(r0 + (80));
              (v113_data + (v77_data * v45_data)).copy_to(r0 + (80));
              int32_t v117_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v124_data;
              v124_data.copy_from(r0 + (106));
              (v124_data + (v77_data * v56_data)).copy_to(r0 + (106));
              int32_t v128_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v135_data;
              v135_data.copy_from(r0 + (132));
              (v135_data + (v77_data * v67_data)).copy_to(r0 + (132));
              int32_t v139_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v143_data;
              v143_data.copy_from(glb_m0 + (128_i32));
              tensorforge::intel_esimd::simd<float, 32> v146_data;
              v146_data.copy_from(r0 + (4));
              (v146_data + (v143_data * v12_data)).copy_to(r0 + (4));
              int32_t v150_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v157_data;
              v157_data.copy_from(r0 + (30));
              (v157_data + (v143_data * v23_data)).copy_to(r0 + (30));
              int32_t v161_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v168_data;
              v168_data.copy_from(r0 + (56));
              (v168_data + (v143_data * v34_data)).copy_to(r0 + (56));
              int32_t v172_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v179_data;
              v179_data.copy_from(r0 + (82));
              (v179_data + (v143_data * v45_data)).copy_to(r0 + (82));
              int32_t v183_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v190_data;
              v190_data.copy_from(r0 + (108));
              (v190_data + (v143_data * v56_data)).copy_to(r0 + (108));
              int32_t v194_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v201_data;
              v201_data.copy_from(r0 + (134));
              (v201_data + (v143_data * v67_data)).copy_to(r0 + (134));
              int32_t v205_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v209_data;
              v209_data.copy_from(glb_m0 + (192_i32));
              tensorforge::intel_esimd::simd<float, 32> v212_data;
              v212_data.copy_from(r0 + (6));
              (v212_data + (v209_data * v12_data)).copy_to(r0 + (6));
              int32_t v216_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v223_data;
              v223_data.copy_from(r0 + (32));
              (v223_data + (v209_data * v23_data)).copy_to(r0 + (32));
              int32_t v227_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v234_data;
              v234_data.copy_from(r0 + (58));
              (v234_data + (v209_data * v34_data)).copy_to(r0 + (58));
              int32_t v238_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v245_data;
              v245_data.copy_from(r0 + (84));
              (v245_data + (v209_data * v45_data)).copy_to(r0 + (84));
              int32_t v249_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v256_data;
              v256_data.copy_from(r0 + (110));
              (v256_data + (v209_data * v56_data)).copy_to(r0 + (110));
              int32_t v260_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v267_data;
              v267_data.copy_from(r0 + (136));
              (v267_data + (v209_data * v67_data)).copy_to(r0 + (136));
              int32_t v271_a = 0_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v275_data;
              v275_data.copy_from(glb_m0 + (256_i32));
              tensorforge::intel_esimd::simd<float, 32> v278_data;
              v278_data.copy_from(r0 + (8));
              (v278_data + (v275_data * v12_data)).copy_to(r0 + (8));
              int32_t v282_a = 0_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v289_data;
              v289_data.copy_from(r0 + (34));
              (v289_data + (v275_data * v23_data)).copy_to(r0 + (34));
              int32_t v293_a = 0_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v300_data;
              v300_data.copy_from(r0 + (60));
              (v300_data + (v275_data * v34_data)).copy_to(r0 + (60));
              int32_t v304_a = 0_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v311_data;
              v311_data.copy_from(r0 + (86));
              (v311_data + (v275_data * v45_data)).copy_to(r0 + (86));
              int32_t v315_a = 0_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v322_data;
              v322_data.copy_from(r0 + (112));
              (v322_data + (v275_data * v56_data)).copy_to(r0 + (112));
              int32_t v326_a = 0_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v333_data;
              v333_data.copy_from(r0 + (138));
              (v333_data + (v275_data * v67_data)).copy_to(r0 + (138));
              int32_t v337_a = 0_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v341_data;
              v341_data.copy_from(glb_m0 + (320_i32));
              tensorforge::intel_esimd::simd<float, 32> v344_data;
              v344_data.copy_from(r0 + (10));
              (v344_data + (v341_data * v12_data)).copy_to(r0 + (10));
              int32_t v348_a = 0_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v355_data;
              v355_data.copy_from(r0 + (36));
              (v355_data + (v341_data * v23_data)).copy_to(r0 + (36));
              int32_t v359_a = 0_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v366_data;
              v366_data.copy_from(r0 + (62));
              (v366_data + (v341_data * v34_data)).copy_to(r0 + (62));
              int32_t v370_a = 0_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v377_data;
              v377_data.copy_from(r0 + (88));
              (v377_data + (v341_data * v45_data)).copy_to(r0 + (88));
              int32_t v381_a = 0_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v388_data;
              v388_data.copy_from(r0 + (114));
              (v388_data + (v341_data * v56_data)).copy_to(r0 + (114));
              int32_t v392_a = 0_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v399_data;
              v399_data.copy_from(r0 + (140));
              (v399_data + (v341_data * v67_data)).copy_to(r0 + (140));
              int32_t v403_a = 0_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v407_data;
              v407_data.copy_from(glb_m0 + (384_i32));
              tensorforge::intel_esimd::simd<float, 32> v410_data;
              v410_data.copy_from(r0 + (12));
              (v410_data + (v407_data * v12_data)).copy_to(r0 + (12));
              int32_t v414_a = 0_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v421_data;
              v421_data.copy_from(r0 + (38));
              (v421_data + (v407_data * v23_data)).copy_to(r0 + (38));
              int32_t v425_a = 0_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v432_data;
              v432_data.copy_from(r0 + (64));
              (v432_data + (v407_data * v34_data)).copy_to(r0 + (64));
              int32_t v436_a = 0_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v443_data;
              v443_data.copy_from(r0 + (90));
              (v443_data + (v407_data * v45_data)).copy_to(r0 + (90));
              int32_t v447_a = 0_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v454_data;
              v454_data.copy_from(r0 + (116));
              (v454_data + (v407_data * v56_data)).copy_to(r0 + (116));
              int32_t v458_a = 0_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v465_data;
              v465_data.copy_from(r0 + (142));
              (v465_data + (v407_data * v67_data)).copy_to(r0 + (142));
              int32_t v469_a = 0_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v473_data;
              v473_data.copy_from(glb_m0 + (448_i32));
              tensorforge::intel_esimd::simd<float, 32> v476_data;
              v476_data.copy_from(r0 + (14));
              (v476_data + (v473_data * v12_data)).copy_to(r0 + (14));
              int32_t v480_a = 0_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v487_data;
              v487_data.copy_from(r0 + (40));
              (v487_data + (v473_data * v23_data)).copy_to(r0 + (40));
              int32_t v491_a = 0_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v498_data;
              v498_data.copy_from(r0 + (66));
              (v498_data + (v473_data * v34_data)).copy_to(r0 + (66));
              int32_t v502_a = 0_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v509_data;
              v509_data.copy_from(r0 + (92));
              (v509_data + (v473_data * v45_data)).copy_to(r0 + (92));
              int32_t v513_a = 0_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v520_data;
              v520_data.copy_from(r0 + (118));
              (v520_data + (v473_data * v56_data)).copy_to(r0 + (118));
              int32_t v524_a = 0_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v531_data;
              v531_data.copy_from(r0 + (144));
              (v531_data + (v473_data * v67_data)).copy_to(r0 + (144));
              int32_t v535_a = 0_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v539_data;
              v539_data.copy_from(glb_m0 + (512_i32));
              tensorforge::intel_esimd::simd<float, 32> v542_data;
              v542_data.copy_from(r0 + (16));
              (v542_data + (v539_data * v12_data)).copy_to(r0 + (16));
              int32_t v546_a = 0_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v553_data;
              v553_data.copy_from(r0 + (42));
              (v553_data + (v539_data * v23_data)).copy_to(r0 + (42));
              int32_t v557_a = 0_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v564_data;
              v564_data.copy_from(r0 + (68));
              (v564_data + (v539_data * v34_data)).copy_to(r0 + (68));
              int32_t v568_a = 0_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v575_data;
              v575_data.copy_from(r0 + (94));
              (v575_data + (v539_data * v45_data)).copy_to(r0 + (94));
              int32_t v579_a = 0_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v586_data;
              v586_data.copy_from(r0 + (120));
              (v586_data + (v539_data * v56_data)).copy_to(r0 + (120));
              int32_t v590_a = 0_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v597_data;
              v597_data.copy_from(r0 + (146));
              (v597_data + (v539_data * v67_data)).copy_to(r0 + (146));
              int32_t v601_a = 0_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v605_data;
              v605_data.copy_from(glb_m0 + (576_i32));
              tensorforge::intel_esimd::simd<float, 32> v608_data;
              v608_data.copy_from(r0 + (18));
              (v608_data + (v605_data * v12_data)).copy_to(r0 + (18));
              int32_t v612_a = 0_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v619_data;
              v619_data.copy_from(r0 + (44));
              (v619_data + (v605_data * v23_data)).copy_to(r0 + (44));
              int32_t v623_a = 0_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v630_data;
              v630_data.copy_from(r0 + (70));
              (v630_data + (v605_data * v34_data)).copy_to(r0 + (70));
              int32_t v634_a = 0_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v641_data;
              v641_data.copy_from(r0 + (96));
              (v641_data + (v605_data * v45_data)).copy_to(r0 + (96));
              int32_t v645_a = 0_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v652_data;
              v652_data.copy_from(r0 + (122));
              (v652_data + (v605_data * v56_data)).copy_to(r0 + (122));
              int32_t v656_a = 0_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v663_data;
              v663_data.copy_from(r0 + (148));
              (v663_data + (v605_data * v67_data)).copy_to(r0 + (148));
              int32_t v667_a = 0_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v671_data;
              v671_data.copy_from(glb_m0 + (640_i32));
              tensorforge::intel_esimd::simd<float, 32> v674_data;
              v674_data.copy_from(r0 + (20));
              (v674_data + (v671_data * v12_data)).copy_to(r0 + (20));
              int32_t v678_a = 0_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v685_data;
              v685_data.copy_from(r0 + (46));
              (v685_data + (v671_data * v23_data)).copy_to(r0 + (46));
              int32_t v689_a = 0_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v696_data;
              v696_data.copy_from(r0 + (72));
              (v696_data + (v671_data * v34_data)).copy_to(r0 + (72));
              int32_t v700_a = 0_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v707_data;
              v707_data.copy_from(r0 + (98));
              (v707_data + (v671_data * v45_data)).copy_to(r0 + (98));
              int32_t v711_a = 0_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v718_data;
              v718_data.copy_from(r0 + (124));
              (v718_data + (v671_data * v56_data)).copy_to(r0 + (124));
              int32_t v722_a = 0_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v729_data;
              v729_data.copy_from(r0 + (150));
              (v729_data + (v671_data * v67_data)).copy_to(r0 + (150));
              int32_t v733_a = 0_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v737_data;
              v737_data.copy_from(glb_m0 + (704_i32));
              tensorforge::intel_esimd::simd<float, 32> v740_data;
              v740_data.copy_from(r0 + (22));
              (v740_data + (v737_data * v12_data)).copy_to(r0 + (22));
              int32_t v744_a = 0_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v751_data;
              v751_data.copy_from(r0 + (48));
              (v751_data + (v737_data * v23_data)).copy_to(r0 + (48));
              int32_t v755_a = 0_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v762_data;
              v762_data.copy_from(r0 + (74));
              (v762_data + (v737_data * v34_data)).copy_to(r0 + (74));
              int32_t v766_a = 0_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v773_data;
              v773_data.copy_from(r0 + (100));
              (v773_data + (v737_data * v45_data)).copy_to(r0 + (100));
              int32_t v777_a = 0_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v784_data;
              v784_data.copy_from(r0 + (126));
              (v784_data + (v737_data * v56_data)).copy_to(r0 + (126));
              int32_t v788_a = 0_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v795_data;
              v795_data.copy_from(r0 + (152));
              (v795_data + (v737_data * v67_data)).copy_to(r0 + (152));
              int32_t v799_a = 0_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v803_data;
              v803_data.copy_from(glb_m0 + (768_i32));
              tensorforge::intel_esimd::simd<float, 32> v806_data;
              v806_data.copy_from(r0 + (24));
              (v806_data + (v803_data * v12_data)).copy_to(r0 + (24));
              int32_t v810_a = 0_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v817_data;
              v817_data.copy_from(r0 + (50));
              (v817_data + (v803_data * v23_data)).copy_to(r0 + (50));
              int32_t v821_a = 0_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v828_data;
              v828_data.copy_from(r0 + (76));
              (v828_data + (v803_data * v34_data)).copy_to(r0 + (76));
              int32_t v832_a = 0_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v839_data;
              v839_data.copy_from(r0 + (102));
              (v839_data + (v803_data * v45_data)).copy_to(r0 + (102));
              int32_t v843_a = 0_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v850_data;
              v850_data.copy_from(r0 + (128));
              (v850_data + (v803_data * v56_data)).copy_to(r0 + (128));
              int32_t v854_a = 0_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v861_data;
              v861_data.copy_from(r0 + (154));
              (v861_data + (v803_data * v67_data)).copy_to(r0 + (154));
              int32_t v865_a = 32_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v869_data;
              v869_data.copy_from(glb_m0 + (32_i32));
              tensorforge::intel_esimd::simd<float, 32> v872_data;
              v872_data.copy_from(r0 + (1));
              (v872_data + (v869_data * v12_data)).copy_to(r0 + (1));
              int32_t v876_a = 32_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v883_data;
              v883_data.copy_from(r0 + (27));
              (v883_data + (v869_data * v23_data)).copy_to(r0 + (27));
              int32_t v887_a = 32_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v894_data;
              v894_data.copy_from(r0 + (53));
              (v894_data + (v869_data * v34_data)).copy_to(r0 + (53));
              int32_t v898_a = 32_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v905_data;
              v905_data.copy_from(r0 + (79));
              (v905_data + (v869_data * v45_data)).copy_to(r0 + (79));
              int32_t v909_a = 32_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v916_data;
              v916_data.copy_from(r0 + (105));
              (v916_data + (v869_data * v56_data)).copy_to(r0 + (105));
              int32_t v920_a = 32_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v927_data;
              v927_data.copy_from(r0 + (131));
              (v927_data + (v869_data * v67_data)).copy_to(r0 + (131));
              int32_t v931_a = 32_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v935_data;
              v935_data.copy_from(glb_m0 + (96_i32));
              tensorforge::intel_esimd::simd<float, 32> v938_data;
              v938_data.copy_from(r0 + (3));
              (v938_data + (v935_data * v12_data)).copy_to(r0 + (3));
              int32_t v942_a = 32_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v949_data;
              v949_data.copy_from(r0 + (29));
              (v949_data + (v935_data * v23_data)).copy_to(r0 + (29));
              int32_t v953_a = 32_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v960_data;
              v960_data.copy_from(r0 + (55));
              (v960_data + (v935_data * v34_data)).copy_to(r0 + (55));
              int32_t v964_a = 32_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v971_data;
              v971_data.copy_from(r0 + (81));
              (v971_data + (v935_data * v45_data)).copy_to(r0 + (81));
              int32_t v975_a = 32_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v982_data;
              v982_data.copy_from(r0 + (107));
              (v982_data + (v935_data * v56_data)).copy_to(r0 + (107));
              int32_t v986_a = 32_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v993_data;
              v993_data.copy_from(r0 + (133));
              (v993_data + (v935_data * v67_data)).copy_to(r0 + (133));
              int32_t v997_a = 32_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v1001_data;
              v1001_data.copy_from(glb_m0 + (160_i32));
              tensorforge::intel_esimd::simd<float, 32> v1004_data;
              v1004_data.copy_from(r0 + (5));
              (v1004_data + (v1001_data * v12_data)).copy_to(r0 + (5));
              int32_t v1008_a = 32_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v1015_data;
              v1015_data.copy_from(r0 + (31));
              (v1015_data + (v1001_data * v23_data)).copy_to(r0 + (31));
              int32_t v1019_a = 32_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v1026_data;
              v1026_data.copy_from(r0 + (57));
              (v1026_data + (v1001_data * v34_data)).copy_to(r0 + (57));
              int32_t v1030_a = 32_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v1037_data;
              v1037_data.copy_from(r0 + (83));
              (v1037_data + (v1001_data * v45_data)).copy_to(r0 + (83));
              int32_t v1041_a = 32_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v1048_data;
              v1048_data.copy_from(r0 + (109));
              (v1048_data + (v1001_data * v56_data)).copy_to(r0 + (109));
              int32_t v1052_a = 32_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v1059_data;
              v1059_data.copy_from(r0 + (135));
              (v1059_data + (v1001_data * v67_data)).copy_to(r0 + (135));
              int32_t v1063_a = 32_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v1067_data;
              v1067_data.copy_from(glb_m0 + (224_i32));
              tensorforge::intel_esimd::simd<float, 32> v1070_data;
              v1070_data.copy_from(r0 + (7));
              (v1070_data + (v1067_data * v12_data)).copy_to(r0 + (7));
              int32_t v1074_a = 32_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v1081_data;
              v1081_data.copy_from(r0 + (33));
              (v1081_data + (v1067_data * v23_data)).copy_to(r0 + (33));
              int32_t v1085_a = 32_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v1092_data;
              v1092_data.copy_from(r0 + (59));
              (v1092_data + (v1067_data * v34_data)).copy_to(r0 + (59));
              int32_t v1096_a = 32_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v1103_data;
              v1103_data.copy_from(r0 + (85));
              (v1103_data + (v1067_data * v45_data)).copy_to(r0 + (85));
              int32_t v1107_a = 32_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v1114_data;
              v1114_data.copy_from(r0 + (111));
              (v1114_data + (v1067_data * v56_data)).copy_to(r0 + (111));
              int32_t v1118_a = 32_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v1125_data;
              v1125_data.copy_from(r0 + (137));
              (v1125_data + (v1067_data * v67_data)).copy_to(r0 + (137));
              int32_t v1129_a = 32_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v1133_data;
              v1133_data.copy_from(glb_m0 + (288_i32));
              tensorforge::intel_esimd::simd<float, 32> v1136_data;
              v1136_data.copy_from(r0 + (9));
              (v1136_data + (v1133_data * v12_data)).copy_to(r0 + (9));
              int32_t v1140_a = 32_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v1147_data;
              v1147_data.copy_from(r0 + (35));
              (v1147_data + (v1133_data * v23_data)).copy_to(r0 + (35));
              int32_t v1151_a = 32_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v1158_data;
              v1158_data.copy_from(r0 + (61));
              (v1158_data + (v1133_data * v34_data)).copy_to(r0 + (61));
              int32_t v1162_a = 32_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v1169_data;
              v1169_data.copy_from(r0 + (87));
              (v1169_data + (v1133_data * v45_data)).copy_to(r0 + (87));
              int32_t v1173_a = 32_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v1180_data;
              v1180_data.copy_from(r0 + (113));
              (v1180_data + (v1133_data * v56_data)).copy_to(r0 + (113));
              int32_t v1184_a = 32_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v1191_data;
              v1191_data.copy_from(r0 + (139));
              (v1191_data + (v1133_data * v67_data)).copy_to(r0 + (139));
              int32_t v1195_a = 32_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v1199_data;
              v1199_data.copy_from(glb_m0 + (352_i32));
              tensorforge::intel_esimd::simd<float, 32> v1202_data;
              v1202_data.copy_from(r0 + (11));
              (v1202_data + (v1199_data * v12_data)).copy_to(r0 + (11));
              int32_t v1206_a = 32_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v1213_data;
              v1213_data.copy_from(r0 + (37));
              (v1213_data + (v1199_data * v23_data)).copy_to(r0 + (37));
              int32_t v1217_a = 32_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v1224_data;
              v1224_data.copy_from(r0 + (63));
              (v1224_data + (v1199_data * v34_data)).copy_to(r0 + (63));
              int32_t v1228_a = 32_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v1235_data;
              v1235_data.copy_from(r0 + (89));
              (v1235_data + (v1199_data * v45_data)).copy_to(r0 + (89));
              int32_t v1239_a = 32_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v1246_data;
              v1246_data.copy_from(r0 + (115));
              (v1246_data + (v1199_data * v56_data)).copy_to(r0 + (115));
              int32_t v1250_a = 32_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v1257_data;
              v1257_data.copy_from(r0 + (141));
              (v1257_data + (v1199_data * v67_data)).copy_to(r0 + (141));
              int32_t v1261_a = 32_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v1265_data;
              v1265_data.copy_from(glb_m0 + (416_i32));
              tensorforge::intel_esimd::simd<float, 32> v1268_data;
              v1268_data.copy_from(r0 + (13));
              (v1268_data + (v1265_data * v12_data)).copy_to(r0 + (13));
              int32_t v1272_a = 32_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v1279_data;
              v1279_data.copy_from(r0 + (39));
              (v1279_data + (v1265_data * v23_data)).copy_to(r0 + (39));
              int32_t v1283_a = 32_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v1290_data;
              v1290_data.copy_from(r0 + (65));
              (v1290_data + (v1265_data * v34_data)).copy_to(r0 + (65));
              int32_t v1294_a = 32_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v1301_data;
              v1301_data.copy_from(r0 + (91));
              (v1301_data + (v1265_data * v45_data)).copy_to(r0 + (91));
              int32_t v1305_a = 32_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v1312_data;
              v1312_data.copy_from(r0 + (117));
              (v1312_data + (v1265_data * v56_data)).copy_to(r0 + (117));
              int32_t v1316_a = 32_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v1323_data;
              v1323_data.copy_from(r0 + (143));
              (v1323_data + (v1265_data * v67_data)).copy_to(r0 + (143));
              int32_t v1327_a = 32_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v1331_data;
              v1331_data.copy_from(glb_m0 + (480_i32));
              tensorforge::intel_esimd::simd<float, 32> v1334_data;
              v1334_data.copy_from(r0 + (15));
              (v1334_data + (v1331_data * v12_data)).copy_to(r0 + (15));
              int32_t v1338_a = 32_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v1345_data;
              v1345_data.copy_from(r0 + (41));
              (v1345_data + (v1331_data * v23_data)).copy_to(r0 + (41));
              int32_t v1349_a = 32_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v1356_data;
              v1356_data.copy_from(r0 + (67));
              (v1356_data + (v1331_data * v34_data)).copy_to(r0 + (67));
              int32_t v1360_a = 32_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v1367_data;
              v1367_data.copy_from(r0 + (93));
              (v1367_data + (v1331_data * v45_data)).copy_to(r0 + (93));
              int32_t v1371_a = 32_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v1378_data;
              v1378_data.copy_from(r0 + (119));
              (v1378_data + (v1331_data * v56_data)).copy_to(r0 + (119));
              int32_t v1382_a = 32_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v1389_data;
              v1389_data.copy_from(r0 + (145));
              (v1389_data + (v1331_data * v67_data)).copy_to(r0 + (145));
              int32_t v1393_a = 32_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v1397_data;
              v1397_data.copy_from(glb_m0 + (544_i32));
              tensorforge::intel_esimd::simd<float, 32> v1400_data;
              v1400_data.copy_from(r0 + (17));
              (v1400_data + (v1397_data * v12_data)).copy_to(r0 + (17));
              int32_t v1404_a = 32_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v1411_data;
              v1411_data.copy_from(r0 + (43));
              (v1411_data + (v1397_data * v23_data)).copy_to(r0 + (43));
              int32_t v1415_a = 32_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v1422_data;
              v1422_data.copy_from(r0 + (69));
              (v1422_data + (v1397_data * v34_data)).copy_to(r0 + (69));
              int32_t v1426_a = 32_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v1433_data;
              v1433_data.copy_from(r0 + (95));
              (v1433_data + (v1397_data * v45_data)).copy_to(r0 + (95));
              int32_t v1437_a = 32_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v1444_data;
              v1444_data.copy_from(r0 + (121));
              (v1444_data + (v1397_data * v56_data)).copy_to(r0 + (121));
              int32_t v1448_a = 32_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v1455_data;
              v1455_data.copy_from(r0 + (147));
              (v1455_data + (v1397_data * v67_data)).copy_to(r0 + (147));
              int32_t v1459_a = 32_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v1463_data;
              v1463_data.copy_from(glb_m0 + (608_i32));
              tensorforge::intel_esimd::simd<float, 32> v1466_data;
              v1466_data.copy_from(r0 + (19));
              (v1466_data + (v1463_data * v12_data)).copy_to(r0 + (19));
              int32_t v1470_a = 32_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v1477_data;
              v1477_data.copy_from(r0 + (45));
              (v1477_data + (v1463_data * v23_data)).copy_to(r0 + (45));
              int32_t v1481_a = 32_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v1488_data;
              v1488_data.copy_from(r0 + (71));
              (v1488_data + (v1463_data * v34_data)).copy_to(r0 + (71));
              int32_t v1492_a = 32_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v1499_data;
              v1499_data.copy_from(r0 + (97));
              (v1499_data + (v1463_data * v45_data)).copy_to(r0 + (97));
              int32_t v1503_a = 32_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v1510_data;
              v1510_data.copy_from(r0 + (123));
              (v1510_data + (v1463_data * v56_data)).copy_to(r0 + (123));
              int32_t v1514_a = 32_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v1521_data;
              v1521_data.copy_from(r0 + (149));
              (v1521_data + (v1463_data * v67_data)).copy_to(r0 + (149));
              int32_t v1525_a = 32_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v1529_data;
              v1529_data.copy_from(glb_m0 + (672_i32));
              tensorforge::intel_esimd::simd<float, 32> v1532_data;
              v1532_data.copy_from(r0 + (21));
              (v1532_data + (v1529_data * v12_data)).copy_to(r0 + (21));
              int32_t v1536_a = 32_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v1543_data;
              v1543_data.copy_from(r0 + (47));
              (v1543_data + (v1529_data * v23_data)).copy_to(r0 + (47));
              int32_t v1547_a = 32_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v1554_data;
              v1554_data.copy_from(r0 + (73));
              (v1554_data + (v1529_data * v34_data)).copy_to(r0 + (73));
              int32_t v1558_a = 32_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v1565_data;
              v1565_data.copy_from(r0 + (99));
              (v1565_data + (v1529_data * v45_data)).copy_to(r0 + (99));
              int32_t v1569_a = 32_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v1576_data;
              v1576_data.copy_from(r0 + (125));
              (v1576_data + (v1529_data * v56_data)).copy_to(r0 + (125));
              int32_t v1580_a = 32_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v1587_data;
              v1587_data.copy_from(r0 + (151));
              (v1587_data + (v1529_data * v67_data)).copy_to(r0 + (151));
              int32_t v1591_a = 32_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v1595_data;
              v1595_data.copy_from(glb_m0 + (736_i32));
              tensorforge::intel_esimd::simd<float, 32> v1598_data;
              v1598_data.copy_from(r0 + (23));
              (v1598_data + (v1595_data * v12_data)).copy_to(r0 + (23));
              int32_t v1602_a = 32_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v1609_data;
              v1609_data.copy_from(r0 + (49));
              (v1609_data + (v1595_data * v23_data)).copy_to(r0 + (49));
              int32_t v1613_a = 32_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v1620_data;
              v1620_data.copy_from(r0 + (75));
              (v1620_data + (v1595_data * v34_data)).copy_to(r0 + (75));
              int32_t v1624_a = 32_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v1631_data;
              v1631_data.copy_from(r0 + (101));
              (v1631_data + (v1595_data * v45_data)).copy_to(r0 + (101));
              int32_t v1635_a = 32_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v1642_data;
              v1642_data.copy_from(r0 + (127));
              (v1642_data + (v1595_data * v56_data)).copy_to(r0 + (127));
              int32_t v1646_a = 32_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v1653_data;
              v1653_data.copy_from(r0 + (153));
              (v1653_data + (v1595_data * v67_data)).copy_to(r0 + (153));
              int32_t v1657_a = 32_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v1661_data;
              v1661_data.copy_from(glb_m0 + (800_i32));
              tensorforge::intel_esimd::simd<float, 32> v1664_data;
              v1664_data.copy_from(r0 + (25));
              (v1664_data + (v1661_data * v12_data)).copy_to(r0 + (25));
              int32_t v1668_a = 32_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v1675_data;
              v1675_data.copy_from(r0 + (51));
              (v1675_data + (v1661_data * v23_data)).copy_to(r0 + (51));
              int32_t v1679_a = 32_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v1686_data;
              v1686_data.copy_from(r0 + (77));
              (v1686_data + (v1661_data * v34_data)).copy_to(r0 + (77));
              int32_t v1690_a = 32_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v1697_data;
              v1697_data.copy_from(r0 + (103));
              (v1697_data + (v1661_data * v45_data)).copy_to(r0 + (103));
              int32_t v1701_a = 32_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v1708_data;
              v1708_data.copy_from(r0 + (129));
              (v1708_data + (v1661_data * v56_data)).copy_to(r0 + (129));
              int32_t v1712_a = 32_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v1719_data;
              v1719_data.copy_from(r0 + (155));
              (v1719_data + (v1661_data * v67_data)).copy_to(r0 + (155));
              float r1[12]{};
              // r1 = +(r0) + name: glb_m2, type: SymbolType.Global, lead: [0]
              // [(20, 35), (0, 1), (0, 6)] []
              float ir1[12]{};
              tensorforge::intel_esimd::simd_mask<32> v1724_g = v4_lead >= 20;
              tensorforge::intel_esimd::simd<float, 32> v1725_data(0.0f);
              v1725_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[24]), v1724_g);
              tensorforge::intel_esimd::simd<float, 32> v1726_data(0.0f);
              v1726_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[0]), v1724_g);
              if (v1724_g) {
                (v1726_data + v1725_data).copy_to(ir1 + (0));
              }
              tensorforge::intel_esimd::simd<float, 32> v1728_data(0.0f);
              v1728_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[50]), v1724_g);
              tensorforge::intel_esimd::simd<float, 32> v1729_data(0.0f);
              v1729_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[2]), v1724_g);
              if (v1724_g) {
                (v1729_data + v1728_data).copy_to(ir1 + (2));
              }
              tensorforge::intel_esimd::simd<float, 32> v1731_data(0.0f);
              v1731_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[76]), v1724_g);
              tensorforge::intel_esimd::simd<float, 32> v1732_data(0.0f);
              v1732_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[4]), v1724_g);
              if (v1724_g) {
                (v1732_data + v1731_data).copy_to(ir1 + (4));
              }
              tensorforge::intel_esimd::simd<float, 32> v1734_data(0.0f);
              v1734_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[102]), v1724_g);
              tensorforge::intel_esimd::simd<float, 32> v1735_data(0.0f);
              v1735_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[6]), v1724_g);
              if (v1724_g) {
                (v1735_data + v1734_data).copy_to(ir1 + (6));
              }
              tensorforge::intel_esimd::simd<float, 32> v1737_data(0.0f);
              v1737_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[128]), v1724_g);
              tensorforge::intel_esimd::simd<float, 32> v1738_data(0.0f);
              v1738_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[8]), v1724_g);
              if (v1724_g) {
                (v1738_data + v1737_data).copy_to(ir1 + (8));
              }
              tensorforge::intel_esimd::simd<float, 32> v1740_data(0.0f);
              v1740_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[154]), v1724_g);
              tensorforge::intel_esimd::simd<float, 32> v1741_data(0.0f);
              v1741_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[10]), v1724_g);
              if (v1724_g) {
                (v1741_data + v1740_data).copy_to(ir1 + (10));
              }
              tensorforge::intel_esimd::simd_mask<32> v1743_g = v4_lead < 3;
              tensorforge::intel_esimd::simd<float, 32> v1744_data(0.0f);
              v1744_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[25]), v1743_g);
              tensorforge::intel_esimd::simd<float, 32> v1745_data(0.0f);
              v1745_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[1]), v1743_g);
              if (v1743_g) {
                (v1745_data + v1744_data).copy_to(ir1 + (1));
              }
              tensorforge::intel_esimd::simd<float, 32> v1747_data(0.0f);
              v1747_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[51]), v1743_g);
              tensorforge::intel_esimd::simd<float, 32> v1748_data(0.0f);
              v1748_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[3]), v1743_g);
              if (v1743_g) {
                (v1748_data + v1747_data).copy_to(ir1 + (3));
              }
              tensorforge::intel_esimd::simd<float, 32> v1750_data(0.0f);
              v1750_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[77]), v1743_g);
              tensorforge::intel_esimd::simd<float, 32> v1751_data(0.0f);
              v1751_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[5]), v1743_g);
              if (v1743_g) {
                (v1751_data + v1750_data).copy_to(ir1 + (5));
              }
              tensorforge::intel_esimd::simd<float, 32> v1753_data(0.0f);
              v1753_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[103]), v1743_g);
              tensorforge::intel_esimd::simd<float, 32> v1754_data(0.0f);
              v1754_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[7]), v1743_g);
              if (v1743_g) {
                (v1754_data + v1753_data).copy_to(ir1 + (7));
              }
              tensorforge::intel_esimd::simd<float, 32> v1756_data(0.0f);
              v1756_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[129]), v1743_g);
              tensorforge::intel_esimd::simd<float, 32> v1757_data(0.0f);
              v1757_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[9]), v1743_g);
              if (v1743_g) {
                (v1757_data + v1756_data).copy_to(ir1 + (9));
              }
              tensorforge::intel_esimd::simd<float, 32> v1759_data(0.0f);
              v1759_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[155]), v1743_g);
              tensorforge::intel_esimd::simd<float, 32> v1760_data(0.0f);
              v1760_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[11]), v1743_g);
              if (v1743_g) {
                (v1760_data + v1759_data).copy_to(ir1 + (11));
              }
              #pragma unroll
              for (int32_t v1764_n1 = 0; v1764_n1 < 1; ++v1764_n1) {
                int32_t v1766_a = v1764_n1 * 2;
                int32_t v1778_a = (v1764_n1 + 12) * 64;
                #pragma unroll
                for (int32_t v1765_n2 = 0; v1765_n2 < 6; ++v1765_n2) {
                  int32_t v1767_a = v1765_n2 * 2;
                  int32_t v1769_a = v1766_a + v1767_a;
                  int32_t v1773_a = v1766_a + v1767_a;
                  tensorforge::intel_esimd::simd<float, 32> v1774_data(0.0f);
                  v1774_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[v1773_a]), v1724_g);
                  int32_t v1779_a = v1765_n2 * 832;
                  int32_t v1781_a = v1778_a + v1779_a;
                  tensorforge::intel_esimd::simd<float, 32> v1789_data(0.0f);
                  v1789_data.merge(tensorforge::intel_esimd::simd<float, 32>(glb_m2[(v1778_a + v1779_a)]), v1724_g);
                  if (v1724_g) {
                    (v1789_data + v1774_data).copy_to(r1 + (v1773_a));
                  }
                }
              }
              #pragma unroll
              for (int32_t v1796_n1 = 0; v1796_n1 < 1; ++v1796_n1) {
                int32_t v1800_a = 1 + (v1796_n1 * 2);
                int32_t v1810_a = (v1796_n1 + 12) * 64;
                int32_t v1812_a = 32_i32 + v1810_a;
                int32_t v1819_a = 32_i32 + v1810_a;
                #pragma unroll
                for (int32_t v1797_n2 = 0; v1797_n2 < 6; ++v1797_n2) {
                  int32_t v1799_a = v1797_n2 * 2;
                  int32_t v1801_a = v1800_a + v1799_a;
                  tensorforge::intel_esimd::simd<float, 32> v1806_data(0.0f);
                  v1806_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[(v1800_a + v1799_a)]), v1743_g);
                  int32_t v1811_a = v1797_n2 * 832;
                  int32_t v1813_a = v1812_a + v1811_a;
                  tensorforge::intel_esimd::simd<float, 32> v1821_data(0.0f);
                  v1821_data.merge(tensorforge::intel_esimd::simd<float, 32>(glb_m2[(v1819_a + v1811_a)]), v1743_g);
                  if (v1743_g) {
                    (v1821_data + v1806_data).copy_to(r1 + ((v1800_a + v1799_a)));
                  }
                }
              }
              // glb_m2 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v1829_i1 = 0; v1829_i1 < 1; ++v1829_i1) {
                int32_t v1831_a = v1829_i1 * 2;
                int32_t v1843_a = (v1829_i1 + 12) * 64;
                #pragma unroll
                for (int32_t v1830_i2 = 0; v1830_i2 < 6; ++v1830_i2) {
                  int32_t v1832_a = v1830_i2 * 2;
                  int32_t v1834_a = v1831_a + v1832_a;
                  tensorforge::intel_esimd::simd<float, 32> v1839_data(0.0f);
                  v1839_data.merge(tensorforge::intel_esimd::simd<float, 32>(r1[(v1831_a + v1832_a)]), v1724_g);
                  if (v1724_g) {
                    v1839_data.copy_to(glb_m2 + ((v1843_a + (v1830_i2 * 832))));
                  }
                }
              }
              #pragma unroll
              for (int32_t v1848_i1 = 0; v1848_i1 < 1; ++v1848_i1) {
                int32_t v1852_a = 1 + (v1848_i1 * 2);
                int32_t v1864_a = 32_i32 + ((v1848_i1 + 12) * 64);
                #pragma unroll
                for (int32_t v1849_i2 = 0; v1849_i2 < 6; ++v1849_i2) {
                  int32_t v1851_a = v1849_i2 * 2;
                  int32_t v1853_a = v1852_a + v1851_a;
                  tensorforge::intel_esimd::simd<float, 32> v1858_data(0.0f);
                  v1858_data.merge(tensorforge::intel_esimd::simd<float, 32>(r1[(v1852_a + v1851_a)]), v1743_g);
                  if (v1743_g) {
                    v1858_data.copy_to(glb_m2 + ((v1864_a + (v1849_i2 * 832))));
                  }
                }
              }
            }
          }
        }
      });
    }
  });
}

