// === base name ===
kernel_08a27dccde

// === header ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08a27dccde(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08a27dccde(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1792, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 9×9(9×9) {0..9}×{0..9} strided
        // m1 9×9(9×9) {0..9}×{0..9} strided
        // m2 9×9(9×9) {0..9}×{0..9} strided
        // m3 ()  scalar
        // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[112 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[96];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              s0[0 + 0 + 1 * item.get_local_id(0) + 64] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 64];
              if (item.get_local_id(0) < 1) {
                s0[0 + 0 + 1 * item.get_local_id(0) + 80] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 80];
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[144]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir0[144]{};
              tensorforge::intel_esimd::simd<float, 9> v9_data;
              v9_data.copy_from(glb_m1 + (0_i32));
              float v10_data = s0[0];
              tensorforge::intel_esimd::simd<float, 9> v12_data;
              v12_data.copy_from(ir0 + (0));
              (v12_data + (v9_data * v10_data)).copy_to(ir0 + (0));
              float v18_data = s0[9];
              tensorforge::intel_esimd::simd<float, 9> v20_data;
              v20_data.copy_from(ir0 + (16));
              (v20_data + (v9_data * v18_data)).copy_to(ir0 + (16));
              float v26_data = s0[18];
              tensorforge::intel_esimd::simd<float, 9> v28_data;
              v28_data.copy_from(ir0 + (32));
              (v28_data + (v9_data * v26_data)).copy_to(ir0 + (32));
              float v34_data = s0[27];
              tensorforge::intel_esimd::simd<float, 9> v36_data;
              v36_data.copy_from(ir0 + (48));
              (v36_data + (v9_data * v34_data)).copy_to(ir0 + (48));
              float v42_data = s0[36];
              tensorforge::intel_esimd::simd<float, 9> v44_data;
              v44_data.copy_from(ir0 + (64));
              (v44_data + (v9_data * v42_data)).copy_to(ir0 + (64));
              float v50_data = s0[45];
              tensorforge::intel_esimd::simd<float, 9> v52_data;
              v52_data.copy_from(ir0 + (80));
              (v52_data + (v9_data * v50_data)).copy_to(ir0 + (80));
              float v58_data = s0[54];
              tensorforge::intel_esimd::simd<float, 9> v60_data;
              v60_data.copy_from(ir0 + (96));
              (v60_data + (v9_data * v58_data)).copy_to(ir0 + (96));
              float v66_data = s0[63];
              tensorforge::intel_esimd::simd<float, 9> v68_data;
              v68_data.copy_from(ir0 + (112));
              (v68_data + (v9_data * v66_data)).copy_to(ir0 + (112));
              float v74_data = s0[72];
              tensorforge::intel_esimd::simd<float, 9> v76_data;
              v76_data.copy_from(ir0 + (128));
              (v76_data + (v9_data * v74_data)).copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 9> v81_data;
              v81_data.copy_from(glb_m1 + (9_i32));
              float v82_data = s0[1];
              tensorforge::intel_esimd::simd<float, 9> v84_data;
              v84_data.copy_from(ir0 + (0));
              (v84_data + (v81_data * v82_data)).copy_to(ir0 + (0));
              float v90_data = s0[10];
              tensorforge::intel_esimd::simd<float, 9> v92_data;
              v92_data.copy_from(ir0 + (16));
              (v92_data + (v81_data * v90_data)).copy_to(ir0 + (16));
              float v98_data = s0[19];
              tensorforge::intel_esimd::simd<float, 9> v100_data;
              v100_data.copy_from(ir0 + (32));
              (v100_data + (v81_data * v98_data)).copy_to(ir0 + (32));
              float v106_data = s0[28];
              tensorforge::intel_esimd::simd<float, 9> v108_data;
              v108_data.copy_from(ir0 + (48));
              (v108_data + (v81_data * v106_data)).copy_to(ir0 + (48));
              float v114_data = s0[37];
              tensorforge::intel_esimd::simd<float, 9> v116_data;
              v116_data.copy_from(ir0 + (64));
              (v116_data + (v81_data * v114_data)).copy_to(ir0 + (64));
              float v122_data = s0[46];
              tensorforge::intel_esimd::simd<float, 9> v124_data;
              v124_data.copy_from(ir0 + (80));
              (v124_data + (v81_data * v122_data)).copy_to(ir0 + (80));
              float v130_data = s0[55];
              tensorforge::intel_esimd::simd<float, 9> v132_data;
              v132_data.copy_from(ir0 + (96));
              (v132_data + (v81_data * v130_data)).copy_to(ir0 + (96));
              float v138_data = s0[64];
              tensorforge::intel_esimd::simd<float, 9> v140_data;
              v140_data.copy_from(ir0 + (112));
              (v140_data + (v81_data * v138_data)).copy_to(ir0 + (112));
              float v146_data = s0[73];
              tensorforge::intel_esimd::simd<float, 9> v148_data;
              v148_data.copy_from(ir0 + (128));
              (v148_data + (v81_data * v146_data)).copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 9> v153_data;
              v153_data.copy_from(glb_m1 + (18_i32));
              float v154_data = s0[2];
              tensorforge::intel_esimd::simd<float, 9> v156_data;
              v156_data.copy_from(ir0 + (0));
              (v156_data + (v153_data * v154_data)).copy_to(ir0 + (0));
              float v162_data = s0[11];
              tensorforge::intel_esimd::simd<float, 9> v164_data;
              v164_data.copy_from(ir0 + (16));
              (v164_data + (v153_data * v162_data)).copy_to(ir0 + (16));
              float v170_data = s0[20];
              tensorforge::intel_esimd::simd<float, 9> v172_data;
              v172_data.copy_from(ir0 + (32));
              (v172_data + (v153_data * v170_data)).copy_to(ir0 + (32));
              float v178_data = s0[29];
              tensorforge::intel_esimd::simd<float, 9> v180_data;
              v180_data.copy_from(ir0 + (48));
              (v180_data + (v153_data * v178_data)).copy_to(ir0 + (48));
              float v186_data = s0[38];
              tensorforge::intel_esimd::simd<float, 9> v188_data;
              v188_data.copy_from(ir0 + (64));
              (v188_data + (v153_data * v186_data)).copy_to(ir0 + (64));
              float v194_data = s0[47];
              tensorforge::intel_esimd::simd<float, 9> v196_data;
              v196_data.copy_from(ir0 + (80));
              (v196_data + (v153_data * v194_data)).copy_to(ir0 + (80));
              float v202_data = s0[56];
              tensorforge::intel_esimd::simd<float, 9> v204_data;
              v204_data.copy_from(ir0 + (96));
              (v204_data + (v153_data * v202_data)).copy_to(ir0 + (96));
              float v210_data = s0[65];
              tensorforge::intel_esimd::simd<float, 9> v212_data;
              v212_data.copy_from(ir0 + (112));
              (v212_data + (v153_data * v210_data)).copy_to(ir0 + (112));
              float v218_data = s0[74];
              tensorforge::intel_esimd::simd<float, 9> v220_data;
              v220_data.copy_from(ir0 + (128));
              (v220_data + (v153_data * v218_data)).copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 9> v225_data;
              v225_data.copy_from(glb_m1 + (27_i32));
              float v226_data = s0[3];
              tensorforge::intel_esimd::simd<float, 9> v228_data;
              v228_data.copy_from(ir0 + (0));
              (v228_data + (v225_data * v226_data)).copy_to(ir0 + (0));
              float v234_data = s0[12];
              tensorforge::intel_esimd::simd<float, 9> v236_data;
              v236_data.copy_from(ir0 + (16));
              (v236_data + (v225_data * v234_data)).copy_to(ir0 + (16));
              float v242_data = s0[21];
              tensorforge::intel_esimd::simd<float, 9> v244_data;
              v244_data.copy_from(ir0 + (32));
              (v244_data + (v225_data * v242_data)).copy_to(ir0 + (32));
              float v250_data = s0[30];
              tensorforge::intel_esimd::simd<float, 9> v252_data;
              v252_data.copy_from(ir0 + (48));
              (v252_data + (v225_data * v250_data)).copy_to(ir0 + (48));
              float v258_data = s0[39];
              tensorforge::intel_esimd::simd<float, 9> v260_data;
              v260_data.copy_from(ir0 + (64));
              (v260_data + (v225_data * v258_data)).copy_to(ir0 + (64));
              float v266_data = s0[48];
              tensorforge::intel_esimd::simd<float, 9> v268_data;
              v268_data.copy_from(ir0 + (80));
              (v268_data + (v225_data * v266_data)).copy_to(ir0 + (80));
              float v274_data = s0[57];
              tensorforge::intel_esimd::simd<float, 9> v276_data;
              v276_data.copy_from(ir0 + (96));
              (v276_data + (v225_data * v274_data)).copy_to(ir0 + (96));
              float v282_data = s0[66];
              tensorforge::intel_esimd::simd<float, 9> v284_data;
              v284_data.copy_from(ir0 + (112));
              (v284_data + (v225_data * v282_data)).copy_to(ir0 + (112));
              float v290_data = s0[75];
              tensorforge::intel_esimd::simd<float, 9> v292_data;
              v292_data.copy_from(ir0 + (128));
              (v292_data + (v225_data * v290_data)).copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 9> v297_data;
              v297_data.copy_from(glb_m1 + (36_i32));
              float v298_data = s0[4];
              tensorforge::intel_esimd::simd<float, 9> v300_data;
              v300_data.copy_from(ir0 + (0));
              (v300_data + (v297_data * v298_data)).copy_to(ir0 + (0));
              float v306_data = s0[13];
              tensorforge::intel_esimd::simd<float, 9> v308_data;
              v308_data.copy_from(ir0 + (16));
              (v308_data + (v297_data * v306_data)).copy_to(ir0 + (16));
              float v314_data = s0[22];
              tensorforge::intel_esimd::simd<float, 9> v316_data;
              v316_data.copy_from(ir0 + (32));
              (v316_data + (v297_data * v314_data)).copy_to(ir0 + (32));
              float v322_data = s0[31];
              tensorforge::intel_esimd::simd<float, 9> v324_data;
              v324_data.copy_from(ir0 + (48));
              (v324_data + (v297_data * v322_data)).copy_to(ir0 + (48));
              float v330_data = s0[40];
              tensorforge::intel_esimd::simd<float, 9> v332_data;
              v332_data.copy_from(ir0 + (64));
              (v332_data + (v297_data * v330_data)).copy_to(ir0 + (64));
              float v338_data = s0[49];
              tensorforge::intel_esimd::simd<float, 9> v340_data;
              v340_data.copy_from(ir0 + (80));
              (v340_data + (v297_data * v338_data)).copy_to(ir0 + (80));
              float v346_data = s0[58];
              tensorforge::intel_esimd::simd<float, 9> v348_data;
              v348_data.copy_from(ir0 + (96));
              (v348_data + (v297_data * v346_data)).copy_to(ir0 + (96));
              float v354_data = s0[67];
              tensorforge::intel_esimd::simd<float, 9> v356_data;
              v356_data.copy_from(ir0 + (112));
              (v356_data + (v297_data * v354_data)).copy_to(ir0 + (112));
              float v362_data = s0[76];
              tensorforge::intel_esimd::simd<float, 9> v364_data;
              v364_data.copy_from(ir0 + (128));
              (v364_data + (v297_data * v362_data)).copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 9> v369_data;
              v369_data.copy_from(glb_m1 + (45_i32));
              float v370_data = s0[5];
              tensorforge::intel_esimd::simd<float, 9> v372_data;
              v372_data.copy_from(ir0 + (0));
              (v372_data + (v369_data * v370_data)).copy_to(ir0 + (0));
              float v378_data = s0[14];
              tensorforge::intel_esimd::simd<float, 9> v380_data;
              v380_data.copy_from(ir0 + (16));
              (v380_data + (v369_data * v378_data)).copy_to(ir0 + (16));
              float v386_data = s0[23];
              tensorforge::intel_esimd::simd<float, 9> v388_data;
              v388_data.copy_from(ir0 + (32));
              (v388_data + (v369_data * v386_data)).copy_to(ir0 + (32));
              float v394_data = s0[32];
              tensorforge::intel_esimd::simd<float, 9> v396_data;
              v396_data.copy_from(ir0 + (48));
              (v396_data + (v369_data * v394_data)).copy_to(ir0 + (48));
              float v402_data = s0[41];
              tensorforge::intel_esimd::simd<float, 9> v404_data;
              v404_data.copy_from(ir0 + (64));
              (v404_data + (v369_data * v402_data)).copy_to(ir0 + (64));
              float v410_data = s0[50];
              tensorforge::intel_esimd::simd<float, 9> v412_data;
              v412_data.copy_from(ir0 + (80));
              (v412_data + (v369_data * v410_data)).copy_to(ir0 + (80));
              float v418_data = s0[59];
              tensorforge::intel_esimd::simd<float, 9> v420_data;
              v420_data.copy_from(ir0 + (96));
              (v420_data + (v369_data * v418_data)).copy_to(ir0 + (96));
              float v426_data = s0[68];
              tensorforge::intel_esimd::simd<float, 9> v428_data;
              v428_data.copy_from(ir0 + (112));
              (v428_data + (v369_data * v426_data)).copy_to(ir0 + (112));
              float v434_data = s0[77];
              tensorforge::intel_esimd::simd<float, 9> v436_data;
              v436_data.copy_from(ir0 + (128));
              (v436_data + (v369_data * v434_data)).copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 9> v441_data;
              v441_data.copy_from(glb_m1 + (54_i32));
              float v442_data = s0[6];
              tensorforge::intel_esimd::simd<float, 9> v444_data;
              v444_data.copy_from(ir0 + (0));
              (v444_data + (v441_data * v442_data)).copy_to(ir0 + (0));
              float v450_data = s0[15];
              tensorforge::intel_esimd::simd<float, 9> v452_data;
              v452_data.copy_from(ir0 + (16));
              (v452_data + (v441_data * v450_data)).copy_to(ir0 + (16));
              float v458_data = s0[24];
              tensorforge::intel_esimd::simd<float, 9> v460_data;
              v460_data.copy_from(ir0 + (32));
              (v460_data + (v441_data * v458_data)).copy_to(ir0 + (32));
              float v466_data = s0[33];
              tensorforge::intel_esimd::simd<float, 9> v468_data;
              v468_data.copy_from(ir0 + (48));
              (v468_data + (v441_data * v466_data)).copy_to(ir0 + (48));
              float v474_data = s0[42];
              tensorforge::intel_esimd::simd<float, 9> v476_data;
              v476_data.copy_from(ir0 + (64));
              (v476_data + (v441_data * v474_data)).copy_to(ir0 + (64));
              float v482_data = s0[51];
              tensorforge::intel_esimd::simd<float, 9> v484_data;
              v484_data.copy_from(ir0 + (80));
              (v484_data + (v441_data * v482_data)).copy_to(ir0 + (80));
              float v490_data = s0[60];
              tensorforge::intel_esimd::simd<float, 9> v492_data;
              v492_data.copy_from(ir0 + (96));
              (v492_data + (v441_data * v490_data)).copy_to(ir0 + (96));
              float v498_data = s0[69];
              tensorforge::intel_esimd::simd<float, 9> v500_data;
              v500_data.copy_from(ir0 + (112));
              (v500_data + (v441_data * v498_data)).copy_to(ir0 + (112));
              float v506_data = s0[78];
              tensorforge::intel_esimd::simd<float, 9> v508_data;
              v508_data.copy_from(ir0 + (128));
              (v508_data + (v441_data * v506_data)).copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 9> v513_data;
              v513_data.copy_from(glb_m1 + (63_i32));
              float v514_data = s0[7];
              tensorforge::intel_esimd::simd<float, 9> v516_data;
              v516_data.copy_from(ir0 + (0));
              (v516_data + (v513_data * v514_data)).copy_to(ir0 + (0));
              float v522_data = s0[16];
              tensorforge::intel_esimd::simd<float, 9> v524_data;
              v524_data.copy_from(ir0 + (16));
              (v524_data + (v513_data * v522_data)).copy_to(ir0 + (16));
              float v530_data = s0[25];
              tensorforge::intel_esimd::simd<float, 9> v532_data;
              v532_data.copy_from(ir0 + (32));
              (v532_data + (v513_data * v530_data)).copy_to(ir0 + (32));
              float v538_data = s0[34];
              tensorforge::intel_esimd::simd<float, 9> v540_data;
              v540_data.copy_from(ir0 + (48));
              (v540_data + (v513_data * v538_data)).copy_to(ir0 + (48));
              float v546_data = s0[43];
              tensorforge::intel_esimd::simd<float, 9> v548_data;
              v548_data.copy_from(ir0 + (64));
              (v548_data + (v513_data * v546_data)).copy_to(ir0 + (64));
              float v554_data = s0[52];
              tensorforge::intel_esimd::simd<float, 9> v556_data;
              v556_data.copy_from(ir0 + (80));
              (v556_data + (v513_data * v554_data)).copy_to(ir0 + (80));
              float v562_data = s0[61];
              tensorforge::intel_esimd::simd<float, 9> v564_data;
              v564_data.copy_from(ir0 + (96));
              (v564_data + (v513_data * v562_data)).copy_to(ir0 + (96));
              float v570_data = s0[70];
              tensorforge::intel_esimd::simd<float, 9> v572_data;
              v572_data.copy_from(ir0 + (112));
              (v572_data + (v513_data * v570_data)).copy_to(ir0 + (112));
              float v578_data = s0[79];
              tensorforge::intel_esimd::simd<float, 9> v580_data;
              v580_data.copy_from(ir0 + (128));
              (v580_data + (v513_data * v578_data)).copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 9> v585_data;
              v585_data.copy_from(glb_m1 + (72_i32));
              float v586_data = s0[8];
              tensorforge::intel_esimd::simd<float, 9> v588_data;
              v588_data.copy_from(ir0 + (0));
              (v588_data + (v585_data * v586_data)).copy_to(ir0 + (0));
              float v594_data = s0[17];
              tensorforge::intel_esimd::simd<float, 9> v596_data;
              v596_data.copy_from(ir0 + (16));
              (v596_data + (v585_data * v594_data)).copy_to(ir0 + (16));
              float v602_data = s0[26];
              tensorforge::intel_esimd::simd<float, 9> v604_data;
              v604_data.copy_from(ir0 + (32));
              (v604_data + (v585_data * v602_data)).copy_to(ir0 + (32));
              float v610_data = s0[35];
              tensorforge::intel_esimd::simd<float, 9> v612_data;
              v612_data.copy_from(ir0 + (48));
              (v612_data + (v585_data * v610_data)).copy_to(ir0 + (48));
              float v618_data = s0[44];
              tensorforge::intel_esimd::simd<float, 9> v620_data;
              v620_data.copy_from(ir0 + (64));
              (v620_data + (v585_data * v618_data)).copy_to(ir0 + (64));
              float v626_data = s0[53];
              tensorforge::intel_esimd::simd<float, 9> v628_data;
              v628_data.copy_from(ir0 + (80));
              (v628_data + (v585_data * v626_data)).copy_to(ir0 + (80));
              float v634_data = s0[62];
              tensorforge::intel_esimd::simd<float, 9> v636_data;
              v636_data.copy_from(ir0 + (96));
              (v636_data + (v585_data * v634_data)).copy_to(ir0 + (96));
              float v642_data = s0[71];
              tensorforge::intel_esimd::simd<float, 9> v644_data;
              v644_data.copy_from(ir0 + (112));
              (v644_data + (v585_data * v642_data)).copy_to(ir0 + (112));
              float v650_data = s0[80];
              tensorforge::intel_esimd::simd<float, 9> v652_data;
              v652_data.copy_from(ir0 + (128));
              (v652_data + (v585_data * v650_data)).copy_to(ir0 + (128));
              #pragma unroll
              for (int32_t v655_n1 = 0; v655_n1 < 9; ++v655_n1) {
                int32_t v656_a = v655_n1 * 16;
                tensorforge::intel_esimd::simd<float, 9> v658_data;
                v658_data.copy_from(ir0 + (v656_a));
                (v658_data * 13.0f).copy_to(r0 + (v656_a));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v662_i1 = 0; v662_i1 < 9; ++v662_i1) {
                tensorforge::intel_esimd::simd<float, 9> v665_data;
                v665_data.copy_from(r0 + ((v662_i1 * 16)));
                v665_data.copy_to(glb_m0 + ((v662_i1 * 9)));
              }
            }
          }
        }
      });
    }
  });
}

