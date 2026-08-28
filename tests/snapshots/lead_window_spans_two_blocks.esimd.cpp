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
              int32_t v6_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v10_data;
              v10_data.copy_from(glb_m0 + (0_i32));
              float v11_data = glb_m1[0];
              tensorforge::intel_esimd::simd<float, 32> v13_data;
              v13_data.copy_from(r0 + (0));
              (v13_data + (v10_data * v11_data)).copy_to(r0 + (0));
              int32_t v17_a = 0_i32 + 0;
              float v22_data = glb_m1[1];
              tensorforge::intel_esimd::simd<float, 32> v24_data;
              v24_data.copy_from(r0 + (26));
              (v24_data + (v10_data * v22_data)).copy_to(r0 + (26));
              int32_t v28_a = 0_i32 + 0;
              float v33_data = glb_m1[2];
              tensorforge::intel_esimd::simd<float, 32> v35_data;
              v35_data.copy_from(r0 + (52));
              (v35_data + (v10_data * v33_data)).copy_to(r0 + (52));
              int32_t v39_a = 0_i32 + 0;
              float v44_data = glb_m1[3];
              tensorforge::intel_esimd::simd<float, 32> v46_data;
              v46_data.copy_from(r0 + (78));
              (v46_data + (v10_data * v44_data)).copy_to(r0 + (78));
              int32_t v50_a = 0_i32 + 0;
              float v55_data = glb_m1[4];
              tensorforge::intel_esimd::simd<float, 32> v57_data;
              v57_data.copy_from(r0 + (104));
              (v57_data + (v10_data * v55_data)).copy_to(r0 + (104));
              int32_t v61_a = 0_i32 + 0;
              float v66_data = glb_m1[5];
              tensorforge::intel_esimd::simd<float, 32> v68_data;
              v68_data.copy_from(r0 + (130));
              (v68_data + (v10_data * v66_data)).copy_to(r0 + (130));
              int32_t v72_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v76_data;
              v76_data.copy_from(glb_m0 + (64_i32));
              tensorforge::intel_esimd::simd<float, 32> v79_data;
              v79_data.copy_from(r0 + (2));
              (v79_data + (v76_data * v11_data)).copy_to(r0 + (2));
              int32_t v83_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v90_data;
              v90_data.copy_from(r0 + (28));
              (v90_data + (v76_data * v22_data)).copy_to(r0 + (28));
              int32_t v94_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v101_data;
              v101_data.copy_from(r0 + (54));
              (v101_data + (v76_data * v33_data)).copy_to(r0 + (54));
              int32_t v105_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v112_data;
              v112_data.copy_from(r0 + (80));
              (v112_data + (v76_data * v44_data)).copy_to(r0 + (80));
              int32_t v116_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v123_data;
              v123_data.copy_from(r0 + (106));
              (v123_data + (v76_data * v55_data)).copy_to(r0 + (106));
              int32_t v127_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v134_data;
              v134_data.copy_from(r0 + (132));
              (v134_data + (v76_data * v66_data)).copy_to(r0 + (132));
              int32_t v138_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v142_data;
              v142_data.copy_from(glb_m0 + (128_i32));
              tensorforge::intel_esimd::simd<float, 32> v145_data;
              v145_data.copy_from(r0 + (4));
              (v145_data + (v142_data * v11_data)).copy_to(r0 + (4));
              int32_t v149_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v156_data;
              v156_data.copy_from(r0 + (30));
              (v156_data + (v142_data * v22_data)).copy_to(r0 + (30));
              int32_t v160_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v167_data;
              v167_data.copy_from(r0 + (56));
              (v167_data + (v142_data * v33_data)).copy_to(r0 + (56));
              int32_t v171_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v178_data;
              v178_data.copy_from(r0 + (82));
              (v178_data + (v142_data * v44_data)).copy_to(r0 + (82));
              int32_t v182_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v189_data;
              v189_data.copy_from(r0 + (108));
              (v189_data + (v142_data * v55_data)).copy_to(r0 + (108));
              int32_t v193_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v200_data;
              v200_data.copy_from(r0 + (134));
              (v200_data + (v142_data * v66_data)).copy_to(r0 + (134));
              int32_t v204_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v208_data;
              v208_data.copy_from(glb_m0 + (192_i32));
              tensorforge::intel_esimd::simd<float, 32> v211_data;
              v211_data.copy_from(r0 + (6));
              (v211_data + (v208_data * v11_data)).copy_to(r0 + (6));
              int32_t v215_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v222_data;
              v222_data.copy_from(r0 + (32));
              (v222_data + (v208_data * v22_data)).copy_to(r0 + (32));
              int32_t v226_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v233_data;
              v233_data.copy_from(r0 + (58));
              (v233_data + (v208_data * v33_data)).copy_to(r0 + (58));
              int32_t v237_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v244_data;
              v244_data.copy_from(r0 + (84));
              (v244_data + (v208_data * v44_data)).copy_to(r0 + (84));
              int32_t v248_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v255_data;
              v255_data.copy_from(r0 + (110));
              (v255_data + (v208_data * v55_data)).copy_to(r0 + (110));
              int32_t v259_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v266_data;
              v266_data.copy_from(r0 + (136));
              (v266_data + (v208_data * v66_data)).copy_to(r0 + (136));
              int32_t v270_a = 0_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v274_data;
              v274_data.copy_from(glb_m0 + (256_i32));
              tensorforge::intel_esimd::simd<float, 32> v277_data;
              v277_data.copy_from(r0 + (8));
              (v277_data + (v274_data * v11_data)).copy_to(r0 + (8));
              int32_t v281_a = 0_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v288_data;
              v288_data.copy_from(r0 + (34));
              (v288_data + (v274_data * v22_data)).copy_to(r0 + (34));
              int32_t v292_a = 0_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v299_data;
              v299_data.copy_from(r0 + (60));
              (v299_data + (v274_data * v33_data)).copy_to(r0 + (60));
              int32_t v303_a = 0_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v310_data;
              v310_data.copy_from(r0 + (86));
              (v310_data + (v274_data * v44_data)).copy_to(r0 + (86));
              int32_t v314_a = 0_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v321_data;
              v321_data.copy_from(r0 + (112));
              (v321_data + (v274_data * v55_data)).copy_to(r0 + (112));
              int32_t v325_a = 0_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v332_data;
              v332_data.copy_from(r0 + (138));
              (v332_data + (v274_data * v66_data)).copy_to(r0 + (138));
              int32_t v336_a = 0_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v340_data;
              v340_data.copy_from(glb_m0 + (320_i32));
              tensorforge::intel_esimd::simd<float, 32> v343_data;
              v343_data.copy_from(r0 + (10));
              (v343_data + (v340_data * v11_data)).copy_to(r0 + (10));
              int32_t v347_a = 0_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v354_data;
              v354_data.copy_from(r0 + (36));
              (v354_data + (v340_data * v22_data)).copy_to(r0 + (36));
              int32_t v358_a = 0_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v365_data;
              v365_data.copy_from(r0 + (62));
              (v365_data + (v340_data * v33_data)).copy_to(r0 + (62));
              int32_t v369_a = 0_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v376_data;
              v376_data.copy_from(r0 + (88));
              (v376_data + (v340_data * v44_data)).copy_to(r0 + (88));
              int32_t v380_a = 0_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v387_data;
              v387_data.copy_from(r0 + (114));
              (v387_data + (v340_data * v55_data)).copy_to(r0 + (114));
              int32_t v391_a = 0_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v398_data;
              v398_data.copy_from(r0 + (140));
              (v398_data + (v340_data * v66_data)).copy_to(r0 + (140));
              int32_t v402_a = 0_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v406_data;
              v406_data.copy_from(glb_m0 + (384_i32));
              tensorforge::intel_esimd::simd<float, 32> v409_data;
              v409_data.copy_from(r0 + (12));
              (v409_data + (v406_data * v11_data)).copy_to(r0 + (12));
              int32_t v413_a = 0_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v420_data;
              v420_data.copy_from(r0 + (38));
              (v420_data + (v406_data * v22_data)).copy_to(r0 + (38));
              int32_t v424_a = 0_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v431_data;
              v431_data.copy_from(r0 + (64));
              (v431_data + (v406_data * v33_data)).copy_to(r0 + (64));
              int32_t v435_a = 0_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v442_data;
              v442_data.copy_from(r0 + (90));
              (v442_data + (v406_data * v44_data)).copy_to(r0 + (90));
              int32_t v446_a = 0_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v453_data;
              v453_data.copy_from(r0 + (116));
              (v453_data + (v406_data * v55_data)).copy_to(r0 + (116));
              int32_t v457_a = 0_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v464_data;
              v464_data.copy_from(r0 + (142));
              (v464_data + (v406_data * v66_data)).copy_to(r0 + (142));
              int32_t v468_a = 0_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v472_data;
              v472_data.copy_from(glb_m0 + (448_i32));
              tensorforge::intel_esimd::simd<float, 32> v475_data;
              v475_data.copy_from(r0 + (14));
              (v475_data + (v472_data * v11_data)).copy_to(r0 + (14));
              int32_t v479_a = 0_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v486_data;
              v486_data.copy_from(r0 + (40));
              (v486_data + (v472_data * v22_data)).copy_to(r0 + (40));
              int32_t v490_a = 0_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v497_data;
              v497_data.copy_from(r0 + (66));
              (v497_data + (v472_data * v33_data)).copy_to(r0 + (66));
              int32_t v501_a = 0_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v508_data;
              v508_data.copy_from(r0 + (92));
              (v508_data + (v472_data * v44_data)).copy_to(r0 + (92));
              int32_t v512_a = 0_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v519_data;
              v519_data.copy_from(r0 + (118));
              (v519_data + (v472_data * v55_data)).copy_to(r0 + (118));
              int32_t v523_a = 0_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v530_data;
              v530_data.copy_from(r0 + (144));
              (v530_data + (v472_data * v66_data)).copy_to(r0 + (144));
              int32_t v534_a = 0_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v538_data;
              v538_data.copy_from(glb_m0 + (512_i32));
              tensorforge::intel_esimd::simd<float, 32> v541_data;
              v541_data.copy_from(r0 + (16));
              (v541_data + (v538_data * v11_data)).copy_to(r0 + (16));
              int32_t v545_a = 0_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v552_data;
              v552_data.copy_from(r0 + (42));
              (v552_data + (v538_data * v22_data)).copy_to(r0 + (42));
              int32_t v556_a = 0_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v563_data;
              v563_data.copy_from(r0 + (68));
              (v563_data + (v538_data * v33_data)).copy_to(r0 + (68));
              int32_t v567_a = 0_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v574_data;
              v574_data.copy_from(r0 + (94));
              (v574_data + (v538_data * v44_data)).copy_to(r0 + (94));
              int32_t v578_a = 0_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v585_data;
              v585_data.copy_from(r0 + (120));
              (v585_data + (v538_data * v55_data)).copy_to(r0 + (120));
              int32_t v589_a = 0_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v596_data;
              v596_data.copy_from(r0 + (146));
              (v596_data + (v538_data * v66_data)).copy_to(r0 + (146));
              int32_t v600_a = 0_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v604_data;
              v604_data.copy_from(glb_m0 + (576_i32));
              tensorforge::intel_esimd::simd<float, 32> v607_data;
              v607_data.copy_from(r0 + (18));
              (v607_data + (v604_data * v11_data)).copy_to(r0 + (18));
              int32_t v611_a = 0_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v618_data;
              v618_data.copy_from(r0 + (44));
              (v618_data + (v604_data * v22_data)).copy_to(r0 + (44));
              int32_t v622_a = 0_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v629_data;
              v629_data.copy_from(r0 + (70));
              (v629_data + (v604_data * v33_data)).copy_to(r0 + (70));
              int32_t v633_a = 0_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v640_data;
              v640_data.copy_from(r0 + (96));
              (v640_data + (v604_data * v44_data)).copy_to(r0 + (96));
              int32_t v644_a = 0_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v651_data;
              v651_data.copy_from(r0 + (122));
              (v651_data + (v604_data * v55_data)).copy_to(r0 + (122));
              int32_t v655_a = 0_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v662_data;
              v662_data.copy_from(r0 + (148));
              (v662_data + (v604_data * v66_data)).copy_to(r0 + (148));
              int32_t v666_a = 0_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v670_data;
              v670_data.copy_from(glb_m0 + (640_i32));
              tensorforge::intel_esimd::simd<float, 32> v673_data;
              v673_data.copy_from(r0 + (20));
              (v673_data + (v670_data * v11_data)).copy_to(r0 + (20));
              int32_t v677_a = 0_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v684_data;
              v684_data.copy_from(r0 + (46));
              (v684_data + (v670_data * v22_data)).copy_to(r0 + (46));
              int32_t v688_a = 0_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v695_data;
              v695_data.copy_from(r0 + (72));
              (v695_data + (v670_data * v33_data)).copy_to(r0 + (72));
              int32_t v699_a = 0_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v706_data;
              v706_data.copy_from(r0 + (98));
              (v706_data + (v670_data * v44_data)).copy_to(r0 + (98));
              int32_t v710_a = 0_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v717_data;
              v717_data.copy_from(r0 + (124));
              (v717_data + (v670_data * v55_data)).copy_to(r0 + (124));
              int32_t v721_a = 0_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v728_data;
              v728_data.copy_from(r0 + (150));
              (v728_data + (v670_data * v66_data)).copy_to(r0 + (150));
              int32_t v732_a = 0_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v736_data;
              v736_data.copy_from(glb_m0 + (704_i32));
              tensorforge::intel_esimd::simd<float, 32> v739_data;
              v739_data.copy_from(r0 + (22));
              (v739_data + (v736_data * v11_data)).copy_to(r0 + (22));
              int32_t v743_a = 0_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v750_data;
              v750_data.copy_from(r0 + (48));
              (v750_data + (v736_data * v22_data)).copy_to(r0 + (48));
              int32_t v754_a = 0_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v761_data;
              v761_data.copy_from(r0 + (74));
              (v761_data + (v736_data * v33_data)).copy_to(r0 + (74));
              int32_t v765_a = 0_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v772_data;
              v772_data.copy_from(r0 + (100));
              (v772_data + (v736_data * v44_data)).copy_to(r0 + (100));
              int32_t v776_a = 0_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v783_data;
              v783_data.copy_from(r0 + (126));
              (v783_data + (v736_data * v55_data)).copy_to(r0 + (126));
              int32_t v787_a = 0_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v794_data;
              v794_data.copy_from(r0 + (152));
              (v794_data + (v736_data * v66_data)).copy_to(r0 + (152));
              int32_t v798_a = 0_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v802_data;
              v802_data.copy_from(glb_m0 + (768_i32));
              tensorforge::intel_esimd::simd<float, 32> v805_data;
              v805_data.copy_from(r0 + (24));
              (v805_data + (v802_data * v11_data)).copy_to(r0 + (24));
              int32_t v809_a = 0_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v816_data;
              v816_data.copy_from(r0 + (50));
              (v816_data + (v802_data * v22_data)).copy_to(r0 + (50));
              int32_t v820_a = 0_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v827_data;
              v827_data.copy_from(r0 + (76));
              (v827_data + (v802_data * v33_data)).copy_to(r0 + (76));
              int32_t v831_a = 0_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v838_data;
              v838_data.copy_from(r0 + (102));
              (v838_data + (v802_data * v44_data)).copy_to(r0 + (102));
              int32_t v842_a = 0_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v849_data;
              v849_data.copy_from(r0 + (128));
              (v849_data + (v802_data * v55_data)).copy_to(r0 + (128));
              int32_t v853_a = 0_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v860_data;
              v860_data.copy_from(r0 + (154));
              (v860_data + (v802_data * v66_data)).copy_to(r0 + (154));
              int32_t v864_a = 32_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v868_data;
              v868_data.copy_from(glb_m0 + (32_i32));
              tensorforge::intel_esimd::simd<float, 32> v871_data;
              v871_data.copy_from(r0 + (1));
              (v871_data + (v868_data * v11_data)).copy_to(r0 + (1));
              int32_t v875_a = 32_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v882_data;
              v882_data.copy_from(r0 + (27));
              (v882_data + (v868_data * v22_data)).copy_to(r0 + (27));
              int32_t v886_a = 32_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v893_data;
              v893_data.copy_from(r0 + (53));
              (v893_data + (v868_data * v33_data)).copy_to(r0 + (53));
              int32_t v897_a = 32_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v904_data;
              v904_data.copy_from(r0 + (79));
              (v904_data + (v868_data * v44_data)).copy_to(r0 + (79));
              int32_t v908_a = 32_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v915_data;
              v915_data.copy_from(r0 + (105));
              (v915_data + (v868_data * v55_data)).copy_to(r0 + (105));
              int32_t v919_a = 32_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v926_data;
              v926_data.copy_from(r0 + (131));
              (v926_data + (v868_data * v66_data)).copy_to(r0 + (131));
              int32_t v930_a = 32_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v934_data;
              v934_data.copy_from(glb_m0 + (96_i32));
              tensorforge::intel_esimd::simd<float, 32> v937_data;
              v937_data.copy_from(r0 + (3));
              (v937_data + (v934_data * v11_data)).copy_to(r0 + (3));
              int32_t v941_a = 32_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v948_data;
              v948_data.copy_from(r0 + (29));
              (v948_data + (v934_data * v22_data)).copy_to(r0 + (29));
              int32_t v952_a = 32_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v959_data;
              v959_data.copy_from(r0 + (55));
              (v959_data + (v934_data * v33_data)).copy_to(r0 + (55));
              int32_t v963_a = 32_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v970_data;
              v970_data.copy_from(r0 + (81));
              (v970_data + (v934_data * v44_data)).copy_to(r0 + (81));
              int32_t v974_a = 32_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v981_data;
              v981_data.copy_from(r0 + (107));
              (v981_data + (v934_data * v55_data)).copy_to(r0 + (107));
              int32_t v985_a = 32_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v992_data;
              v992_data.copy_from(r0 + (133));
              (v992_data + (v934_data * v66_data)).copy_to(r0 + (133));
              int32_t v996_a = 32_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v1000_data;
              v1000_data.copy_from(glb_m0 + (160_i32));
              tensorforge::intel_esimd::simd<float, 32> v1003_data;
              v1003_data.copy_from(r0 + (5));
              (v1003_data + (v1000_data * v11_data)).copy_to(r0 + (5));
              int32_t v1007_a = 32_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v1014_data;
              v1014_data.copy_from(r0 + (31));
              (v1014_data + (v1000_data * v22_data)).copy_to(r0 + (31));
              int32_t v1018_a = 32_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v1025_data;
              v1025_data.copy_from(r0 + (57));
              (v1025_data + (v1000_data * v33_data)).copy_to(r0 + (57));
              int32_t v1029_a = 32_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v1036_data;
              v1036_data.copy_from(r0 + (83));
              (v1036_data + (v1000_data * v44_data)).copy_to(r0 + (83));
              int32_t v1040_a = 32_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v1047_data;
              v1047_data.copy_from(r0 + (109));
              (v1047_data + (v1000_data * v55_data)).copy_to(r0 + (109));
              int32_t v1051_a = 32_i32 + 128;
              tensorforge::intel_esimd::simd<float, 32> v1058_data;
              v1058_data.copy_from(r0 + (135));
              (v1058_data + (v1000_data * v66_data)).copy_to(r0 + (135));
              int32_t v1062_a = 32_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v1066_data;
              v1066_data.copy_from(glb_m0 + (224_i32));
              tensorforge::intel_esimd::simd<float, 32> v1069_data;
              v1069_data.copy_from(r0 + (7));
              (v1069_data + (v1066_data * v11_data)).copy_to(r0 + (7));
              int32_t v1073_a = 32_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v1080_data;
              v1080_data.copy_from(r0 + (33));
              (v1080_data + (v1066_data * v22_data)).copy_to(r0 + (33));
              int32_t v1084_a = 32_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v1091_data;
              v1091_data.copy_from(r0 + (59));
              (v1091_data + (v1066_data * v33_data)).copy_to(r0 + (59));
              int32_t v1095_a = 32_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v1102_data;
              v1102_data.copy_from(r0 + (85));
              (v1102_data + (v1066_data * v44_data)).copy_to(r0 + (85));
              int32_t v1106_a = 32_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v1113_data;
              v1113_data.copy_from(r0 + (111));
              (v1113_data + (v1066_data * v55_data)).copy_to(r0 + (111));
              int32_t v1117_a = 32_i32 + 192;
              tensorforge::intel_esimd::simd<float, 32> v1124_data;
              v1124_data.copy_from(r0 + (137));
              (v1124_data + (v1066_data * v66_data)).copy_to(r0 + (137));
              int32_t v1128_a = 32_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v1132_data;
              v1132_data.copy_from(glb_m0 + (288_i32));
              tensorforge::intel_esimd::simd<float, 32> v1135_data;
              v1135_data.copy_from(r0 + (9));
              (v1135_data + (v1132_data * v11_data)).copy_to(r0 + (9));
              int32_t v1139_a = 32_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v1146_data;
              v1146_data.copy_from(r0 + (35));
              (v1146_data + (v1132_data * v22_data)).copy_to(r0 + (35));
              int32_t v1150_a = 32_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v1157_data;
              v1157_data.copy_from(r0 + (61));
              (v1157_data + (v1132_data * v33_data)).copy_to(r0 + (61));
              int32_t v1161_a = 32_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v1168_data;
              v1168_data.copy_from(r0 + (87));
              (v1168_data + (v1132_data * v44_data)).copy_to(r0 + (87));
              int32_t v1172_a = 32_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v1179_data;
              v1179_data.copy_from(r0 + (113));
              (v1179_data + (v1132_data * v55_data)).copy_to(r0 + (113));
              int32_t v1183_a = 32_i32 + 256;
              tensorforge::intel_esimd::simd<float, 32> v1190_data;
              v1190_data.copy_from(r0 + (139));
              (v1190_data + (v1132_data * v66_data)).copy_to(r0 + (139));
              int32_t v1194_a = 32_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v1198_data;
              v1198_data.copy_from(glb_m0 + (352_i32));
              tensorforge::intel_esimd::simd<float, 32> v1201_data;
              v1201_data.copy_from(r0 + (11));
              (v1201_data + (v1198_data * v11_data)).copy_to(r0 + (11));
              int32_t v1205_a = 32_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v1212_data;
              v1212_data.copy_from(r0 + (37));
              (v1212_data + (v1198_data * v22_data)).copy_to(r0 + (37));
              int32_t v1216_a = 32_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v1223_data;
              v1223_data.copy_from(r0 + (63));
              (v1223_data + (v1198_data * v33_data)).copy_to(r0 + (63));
              int32_t v1227_a = 32_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v1234_data;
              v1234_data.copy_from(r0 + (89));
              (v1234_data + (v1198_data * v44_data)).copy_to(r0 + (89));
              int32_t v1238_a = 32_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v1245_data;
              v1245_data.copy_from(r0 + (115));
              (v1245_data + (v1198_data * v55_data)).copy_to(r0 + (115));
              int32_t v1249_a = 32_i32 + 320;
              tensorforge::intel_esimd::simd<float, 32> v1256_data;
              v1256_data.copy_from(r0 + (141));
              (v1256_data + (v1198_data * v66_data)).copy_to(r0 + (141));
              int32_t v1260_a = 32_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v1264_data;
              v1264_data.copy_from(glb_m0 + (416_i32));
              tensorforge::intel_esimd::simd<float, 32> v1267_data;
              v1267_data.copy_from(r0 + (13));
              (v1267_data + (v1264_data * v11_data)).copy_to(r0 + (13));
              int32_t v1271_a = 32_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v1278_data;
              v1278_data.copy_from(r0 + (39));
              (v1278_data + (v1264_data * v22_data)).copy_to(r0 + (39));
              int32_t v1282_a = 32_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v1289_data;
              v1289_data.copy_from(r0 + (65));
              (v1289_data + (v1264_data * v33_data)).copy_to(r0 + (65));
              int32_t v1293_a = 32_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v1300_data;
              v1300_data.copy_from(r0 + (91));
              (v1300_data + (v1264_data * v44_data)).copy_to(r0 + (91));
              int32_t v1304_a = 32_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v1311_data;
              v1311_data.copy_from(r0 + (117));
              (v1311_data + (v1264_data * v55_data)).copy_to(r0 + (117));
              int32_t v1315_a = 32_i32 + 384;
              tensorforge::intel_esimd::simd<float, 32> v1322_data;
              v1322_data.copy_from(r0 + (143));
              (v1322_data + (v1264_data * v66_data)).copy_to(r0 + (143));
              int32_t v1326_a = 32_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v1330_data;
              v1330_data.copy_from(glb_m0 + (480_i32));
              tensorforge::intel_esimd::simd<float, 32> v1333_data;
              v1333_data.copy_from(r0 + (15));
              (v1333_data + (v1330_data * v11_data)).copy_to(r0 + (15));
              int32_t v1337_a = 32_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v1344_data;
              v1344_data.copy_from(r0 + (41));
              (v1344_data + (v1330_data * v22_data)).copy_to(r0 + (41));
              int32_t v1348_a = 32_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v1355_data;
              v1355_data.copy_from(r0 + (67));
              (v1355_data + (v1330_data * v33_data)).copy_to(r0 + (67));
              int32_t v1359_a = 32_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v1366_data;
              v1366_data.copy_from(r0 + (93));
              (v1366_data + (v1330_data * v44_data)).copy_to(r0 + (93));
              int32_t v1370_a = 32_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v1377_data;
              v1377_data.copy_from(r0 + (119));
              (v1377_data + (v1330_data * v55_data)).copy_to(r0 + (119));
              int32_t v1381_a = 32_i32 + 448;
              tensorforge::intel_esimd::simd<float, 32> v1388_data;
              v1388_data.copy_from(r0 + (145));
              (v1388_data + (v1330_data * v66_data)).copy_to(r0 + (145));
              int32_t v1392_a = 32_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v1396_data;
              v1396_data.copy_from(glb_m0 + (544_i32));
              tensorforge::intel_esimd::simd<float, 32> v1399_data;
              v1399_data.copy_from(r0 + (17));
              (v1399_data + (v1396_data * v11_data)).copy_to(r0 + (17));
              int32_t v1403_a = 32_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v1410_data;
              v1410_data.copy_from(r0 + (43));
              (v1410_data + (v1396_data * v22_data)).copy_to(r0 + (43));
              int32_t v1414_a = 32_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v1421_data;
              v1421_data.copy_from(r0 + (69));
              (v1421_data + (v1396_data * v33_data)).copy_to(r0 + (69));
              int32_t v1425_a = 32_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v1432_data;
              v1432_data.copy_from(r0 + (95));
              (v1432_data + (v1396_data * v44_data)).copy_to(r0 + (95));
              int32_t v1436_a = 32_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v1443_data;
              v1443_data.copy_from(r0 + (121));
              (v1443_data + (v1396_data * v55_data)).copy_to(r0 + (121));
              int32_t v1447_a = 32_i32 + 512;
              tensorforge::intel_esimd::simd<float, 32> v1454_data;
              v1454_data.copy_from(r0 + (147));
              (v1454_data + (v1396_data * v66_data)).copy_to(r0 + (147));
              int32_t v1458_a = 32_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v1462_data;
              v1462_data.copy_from(glb_m0 + (608_i32));
              tensorforge::intel_esimd::simd<float, 32> v1465_data;
              v1465_data.copy_from(r0 + (19));
              (v1465_data + (v1462_data * v11_data)).copy_to(r0 + (19));
              int32_t v1469_a = 32_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v1476_data;
              v1476_data.copy_from(r0 + (45));
              (v1476_data + (v1462_data * v22_data)).copy_to(r0 + (45));
              int32_t v1480_a = 32_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v1487_data;
              v1487_data.copy_from(r0 + (71));
              (v1487_data + (v1462_data * v33_data)).copy_to(r0 + (71));
              int32_t v1491_a = 32_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v1498_data;
              v1498_data.copy_from(r0 + (97));
              (v1498_data + (v1462_data * v44_data)).copy_to(r0 + (97));
              int32_t v1502_a = 32_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v1509_data;
              v1509_data.copy_from(r0 + (123));
              (v1509_data + (v1462_data * v55_data)).copy_to(r0 + (123));
              int32_t v1513_a = 32_i32 + 576;
              tensorforge::intel_esimd::simd<float, 32> v1520_data;
              v1520_data.copy_from(r0 + (149));
              (v1520_data + (v1462_data * v66_data)).copy_to(r0 + (149));
              int32_t v1524_a = 32_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v1528_data;
              v1528_data.copy_from(glb_m0 + (672_i32));
              tensorforge::intel_esimd::simd<float, 32> v1531_data;
              v1531_data.copy_from(r0 + (21));
              (v1531_data + (v1528_data * v11_data)).copy_to(r0 + (21));
              int32_t v1535_a = 32_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v1542_data;
              v1542_data.copy_from(r0 + (47));
              (v1542_data + (v1528_data * v22_data)).copy_to(r0 + (47));
              int32_t v1546_a = 32_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v1553_data;
              v1553_data.copy_from(r0 + (73));
              (v1553_data + (v1528_data * v33_data)).copy_to(r0 + (73));
              int32_t v1557_a = 32_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v1564_data;
              v1564_data.copy_from(r0 + (99));
              (v1564_data + (v1528_data * v44_data)).copy_to(r0 + (99));
              int32_t v1568_a = 32_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v1575_data;
              v1575_data.copy_from(r0 + (125));
              (v1575_data + (v1528_data * v55_data)).copy_to(r0 + (125));
              int32_t v1579_a = 32_i32 + 640;
              tensorforge::intel_esimd::simd<float, 32> v1586_data;
              v1586_data.copy_from(r0 + (151));
              (v1586_data + (v1528_data * v66_data)).copy_to(r0 + (151));
              int32_t v1590_a = 32_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v1594_data;
              v1594_data.copy_from(glb_m0 + (736_i32));
              tensorforge::intel_esimd::simd<float, 32> v1597_data;
              v1597_data.copy_from(r0 + (23));
              (v1597_data + (v1594_data * v11_data)).copy_to(r0 + (23));
              int32_t v1601_a = 32_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v1608_data;
              v1608_data.copy_from(r0 + (49));
              (v1608_data + (v1594_data * v22_data)).copy_to(r0 + (49));
              int32_t v1612_a = 32_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v1619_data;
              v1619_data.copy_from(r0 + (75));
              (v1619_data + (v1594_data * v33_data)).copy_to(r0 + (75));
              int32_t v1623_a = 32_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v1630_data;
              v1630_data.copy_from(r0 + (101));
              (v1630_data + (v1594_data * v44_data)).copy_to(r0 + (101));
              int32_t v1634_a = 32_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v1641_data;
              v1641_data.copy_from(r0 + (127));
              (v1641_data + (v1594_data * v55_data)).copy_to(r0 + (127));
              int32_t v1645_a = 32_i32 + 704;
              tensorforge::intel_esimd::simd<float, 32> v1652_data;
              v1652_data.copy_from(r0 + (153));
              (v1652_data + (v1594_data * v66_data)).copy_to(r0 + (153));
              int32_t v1656_a = 32_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v1660_data;
              v1660_data.copy_from(glb_m0 + (800_i32));
              tensorforge::intel_esimd::simd<float, 32> v1663_data;
              v1663_data.copy_from(r0 + (25));
              (v1663_data + (v1660_data * v11_data)).copy_to(r0 + (25));
              int32_t v1667_a = 32_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v1674_data;
              v1674_data.copy_from(r0 + (51));
              (v1674_data + (v1660_data * v22_data)).copy_to(r0 + (51));
              int32_t v1678_a = 32_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v1685_data;
              v1685_data.copy_from(r0 + (77));
              (v1685_data + (v1660_data * v33_data)).copy_to(r0 + (77));
              int32_t v1689_a = 32_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v1696_data;
              v1696_data.copy_from(r0 + (103));
              (v1696_data + (v1660_data * v44_data)).copy_to(r0 + (103));
              int32_t v1700_a = 32_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v1707_data;
              v1707_data.copy_from(r0 + (129));
              (v1707_data + (v1660_data * v55_data)).copy_to(r0 + (129));
              int32_t v1711_a = 32_i32 + 768;
              tensorforge::intel_esimd::simd<float, 32> v1718_data;
              v1718_data.copy_from(r0 + (155));
              (v1718_data + (v1660_data * v66_data)).copy_to(r0 + (155));
              float r1[12]{};
              // r1 = +(r0) + name: glb_m2, type: SymbolType.Global, lead: [0]
              // [(20, 35), (0, 1), (0, 6)] []
              float ir1[12]{};
              tensorforge::intel_esimd::simd<int32_t, 32> v1722_lead = tensorforge::intel_esimd::simd<int32_t, 32>(0, 1);
              tensorforge::intel_esimd::simd_mask<32> v1723_g = v1722_lead >= 20;
              tensorforge::intel_esimd::simd<float, 32> v1724_data(0.0f);
              v1724_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[24]), v1723_g);
              tensorforge::intel_esimd::simd<float, 32> v1725_data(0.0f);
              v1725_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[0]), v1723_g);
              if (v1723_g) {
                (v1725_data + v1724_data).copy_to(ir1 + (0));
              }
              tensorforge::intel_esimd::simd<float, 32> v1727_data(0.0f);
              v1727_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[50]), v1723_g);
              tensorforge::intel_esimd::simd<float, 32> v1728_data(0.0f);
              v1728_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[2]), v1723_g);
              if (v1723_g) {
                (v1728_data + v1727_data).copy_to(ir1 + (2));
              }
              tensorforge::intel_esimd::simd<float, 32> v1730_data(0.0f);
              v1730_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[76]), v1723_g);
              tensorforge::intel_esimd::simd<float, 32> v1731_data(0.0f);
              v1731_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[4]), v1723_g);
              if (v1723_g) {
                (v1731_data + v1730_data).copy_to(ir1 + (4));
              }
              tensorforge::intel_esimd::simd<float, 32> v1733_data(0.0f);
              v1733_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[102]), v1723_g);
              tensorforge::intel_esimd::simd<float, 32> v1734_data(0.0f);
              v1734_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[6]), v1723_g);
              if (v1723_g) {
                (v1734_data + v1733_data).copy_to(ir1 + (6));
              }
              tensorforge::intel_esimd::simd<float, 32> v1736_data(0.0f);
              v1736_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[128]), v1723_g);
              tensorforge::intel_esimd::simd<float, 32> v1737_data(0.0f);
              v1737_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[8]), v1723_g);
              if (v1723_g) {
                (v1737_data + v1736_data).copy_to(ir1 + (8));
              }
              tensorforge::intel_esimd::simd<float, 32> v1739_data(0.0f);
              v1739_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[154]), v1723_g);
              tensorforge::intel_esimd::simd<float, 32> v1740_data(0.0f);
              v1740_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[10]), v1723_g);
              if (v1723_g) {
                (v1740_data + v1739_data).copy_to(ir1 + (10));
              }
              tensorforge::intel_esimd::simd_mask<32> v1742_g = v1722_lead < 3;
              tensorforge::intel_esimd::simd<float, 32> v1743_data(0.0f);
              v1743_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[25]), v1742_g);
              tensorforge::intel_esimd::simd<float, 32> v1744_data(0.0f);
              v1744_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[1]), v1742_g);
              if (v1742_g) {
                (v1744_data + v1743_data).copy_to(ir1 + (1));
              }
              tensorforge::intel_esimd::simd<float, 32> v1746_data(0.0f);
              v1746_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[51]), v1742_g);
              tensorforge::intel_esimd::simd<float, 32> v1747_data(0.0f);
              v1747_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[3]), v1742_g);
              if (v1742_g) {
                (v1747_data + v1746_data).copy_to(ir1 + (3));
              }
              tensorforge::intel_esimd::simd<float, 32> v1749_data(0.0f);
              v1749_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[77]), v1742_g);
              tensorforge::intel_esimd::simd<float, 32> v1750_data(0.0f);
              v1750_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[5]), v1742_g);
              if (v1742_g) {
                (v1750_data + v1749_data).copy_to(ir1 + (5));
              }
              tensorforge::intel_esimd::simd<float, 32> v1752_data(0.0f);
              v1752_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[103]), v1742_g);
              tensorforge::intel_esimd::simd<float, 32> v1753_data(0.0f);
              v1753_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[7]), v1742_g);
              if (v1742_g) {
                (v1753_data + v1752_data).copy_to(ir1 + (7));
              }
              tensorforge::intel_esimd::simd<float, 32> v1755_data(0.0f);
              v1755_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[129]), v1742_g);
              tensorforge::intel_esimd::simd<float, 32> v1756_data(0.0f);
              v1756_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[9]), v1742_g);
              if (v1742_g) {
                (v1756_data + v1755_data).copy_to(ir1 + (9));
              }
              tensorforge::intel_esimd::simd<float, 32> v1758_data(0.0f);
              v1758_data.merge(tensorforge::intel_esimd::simd<float, 32>(r0[155]), v1742_g);
              tensorforge::intel_esimd::simd<float, 32> v1759_data(0.0f);
              v1759_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[11]), v1742_g);
              if (v1742_g) {
                (v1759_data + v1758_data).copy_to(ir1 + (11));
              }
              #pragma unroll
              for (int32_t v1763_n1 = 0; v1763_n1 < 1; ++v1763_n1) {
                int32_t v1765_a = v1763_n1 * 2;
                int32_t v1777_a = (v1763_n1 + 12) * 64;
                #pragma unroll
                for (int32_t v1764_n2 = 0; v1764_n2 < 6; ++v1764_n2) {
                  int32_t v1766_a = v1764_n2 * 2;
                  int32_t v1768_a = v1765_a + v1766_a;
                  int32_t v1772_a = v1765_a + v1766_a;
                  tensorforge::intel_esimd::simd<float, 32> v1773_data(0.0f);
                  v1773_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[v1772_a]), v1723_g);
                  int32_t v1778_a = v1764_n2 * 832;
                  int32_t v1780_a = v1777_a + v1778_a;
                  tensorforge::intel_esimd::simd<float, 32> v1788_data(0.0f);
                  v1788_data.merge(tensorforge::intel_esimd::simd<float, 32>(glb_m2[(v1777_a + v1778_a)]), v1723_g);
                  if (v1723_g) {
                    (v1788_data + v1773_data).copy_to(r1 + (v1772_a));
                  }
                }
              }
              #pragma unroll
              for (int32_t v1795_n1 = 0; v1795_n1 < 1; ++v1795_n1) {
                int32_t v1799_a = 1 + (v1795_n1 * 2);
                int32_t v1809_a = (v1795_n1 + 12) * 64;
                int32_t v1811_a = 32_i32 + v1809_a;
                int32_t v1818_a = 32_i32 + v1809_a;
                #pragma unroll
                for (int32_t v1796_n2 = 0; v1796_n2 < 6; ++v1796_n2) {
                  int32_t v1798_a = v1796_n2 * 2;
                  int32_t v1800_a = v1799_a + v1798_a;
                  tensorforge::intel_esimd::simd<float, 32> v1805_data(0.0f);
                  v1805_data.merge(tensorforge::intel_esimd::simd<float, 32>(ir1[(v1799_a + v1798_a)]), v1742_g);
                  int32_t v1810_a = v1796_n2 * 832;
                  int32_t v1812_a = v1811_a + v1810_a;
                  tensorforge::intel_esimd::simd<float, 32> v1820_data(0.0f);
                  v1820_data.merge(tensorforge::intel_esimd::simd<float, 32>(glb_m2[(v1818_a + v1810_a)]), v1742_g);
                  if (v1742_g) {
                    (v1820_data + v1805_data).copy_to(r1 + ((v1799_a + v1798_a)));
                  }
                }
              }
              // glb_m2 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v1828_i1 = 0; v1828_i1 < 1; ++v1828_i1) {
                int32_t v1830_a = v1828_i1 * 2;
                int32_t v1842_a = (v1828_i1 + 12) * 64;
                #pragma unroll
                for (int32_t v1829_i2 = 0; v1829_i2 < 6; ++v1829_i2) {
                  int32_t v1831_a = v1829_i2 * 2;
                  int32_t v1833_a = v1830_a + v1831_a;
                  tensorforge::intel_esimd::simd<float, 32> v1838_data(0.0f);
                  v1838_data.merge(tensorforge::intel_esimd::simd<float, 32>(r1[(v1830_a + v1831_a)]), v1723_g);
                  if (v1723_g) {
                    v1838_data.copy_to(glb_m2 + ((v1842_a + (v1829_i2 * 832))));
                  }
                }
              }
              #pragma unroll
              for (int32_t v1847_i1 = 0; v1847_i1 < 1; ++v1847_i1) {
                int32_t v1851_a = 1 + (v1847_i1 * 2);
                int32_t v1863_a = 32_i32 + ((v1847_i1 + 12) * 64);
                #pragma unroll
                for (int32_t v1848_i2 = 0; v1848_i2 < 6; ++v1848_i2) {
                  int32_t v1850_a = v1848_i2 * 2;
                  int32_t v1852_a = v1851_a + v1850_a;
                  tensorforge::intel_esimd::simd<float, 32> v1857_data(0.0f);
                  v1857_data.merge(tensorforge::intel_esimd::simd<float, 32>(r1[(v1851_a + v1850_a)]), v1742_g);
                  if (v1742_g) {
                    v1857_data.copy_to(glb_m2 + ((v1863_a + (v1848_i2 * 832))));
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

