// === base name ===
kernel_ead773dd51

// === header ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ead773dd51(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ead773dd51(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 32×16(32×16) {0..32}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(32×16) {0..32}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 512 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 12) {
                int32_t v16_off = v8_lead + 4;
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 16; ++v10_i1) {
                  float v19_data = glb_m1[(v16_off + (v10_i1 * 32))];
                  r0[v10_i1] = v19_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              float v22_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v22_lin;
              float v23_lin = glb_m2[16 + item.get_local_id(0) * 1];
              r1[1] = v23_lin;
              float v24_lin = glb_m2[32 + item.get_local_id(0) * 1];
              r1[2] = v24_lin;
              float v25_lin = glb_m2[48 + item.get_local_id(0) * 1];
              r1[3] = v25_lin;
              float v26_lin = glb_m2[64 + item.get_local_id(0) * 1];
              r1[4] = v26_lin;
              float v27_lin = glb_m2[80 + item.get_local_id(0) * 1];
              r1[5] = v27_lin;
              float v28_lin = glb_m2[96 + item.get_local_id(0) * 1];
              r1[6] = v28_lin;
              float v29_lin = glb_m2[112 + item.get_local_id(0) * 1];
              r1[7] = v29_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir2[8]{};
              if (v8_lead < 12) {
                float v36_data = r0[0];
                float v37_data = r1[0];
                float v40_data = ir2[0];
                ir2[0] = (v40_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v37_data, 0))));
                float v43_data = r1[1];
                float v46_data = ir2[1];
                ir2[1] = (v46_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 0))));
                float v49_data = r1[2];
                float v52_data = ir2[2];
                ir2[2] = (v52_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 0))));
                float v55_data = r1[3];
                float v58_data = ir2[3];
                ir2[3] = (v58_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 0))));
                float v61_data = r1[4];
                float v64_data = ir2[4];
                ir2[4] = (v64_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 0))));
                float v67_data = r1[5];
                float v70_data = ir2[5];
                ir2[5] = (v70_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
                float v73_data = r1[6];
                float v76_data = ir2[6];
                ir2[6] = (v76_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
                float v79_data = r1[7];
                float v82_data = ir2[7];
                ir2[7] = (v82_data + (v36_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 0))));
              }
              if (v8_lead < 12) {
                float v88_data = r0[1];
                float v89_data = r1[0];
                float v92_data = ir2[0];
                ir2[0] = (v92_data + (v88_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 1))));
                float v95_data = r1[1];
                float v98_data = ir2[1];
                ir2[1] = (v98_data + (v88_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 1))));
                float v101_data = r1[2];
                float v104_data = ir2[2];
                ir2[2] = (v104_data + (v88_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 1))));
                float v107_data = r1[3];
                float v110_data = ir2[3];
                ir2[3] = (v110_data + (v88_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 1))));
                float v113_data = r1[4];
                float v116_data = ir2[4];
                ir2[4] = (v116_data + (v88_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 1))));
                float v119_data = r1[5];
                float v122_data = ir2[5];
                ir2[5] = (v122_data + (v88_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 1))));
                float v125_data = r1[6];
                float v128_data = ir2[6];
                ir2[6] = (v128_data + (v88_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 1))));
                float v131_data = r1[7];
                float v134_data = ir2[7];
                ir2[7] = (v134_data + (v88_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 1))));
              }
              if (v8_lead < 12) {
                float v140_data = r0[2];
                float v141_data = r1[0];
                float v144_data = ir2[0];
                ir2[0] = (v144_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 2))));
                float v147_data = r1[1];
                float v150_data = ir2[1];
                ir2[1] = (v150_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 2))));
                float v153_data = r1[2];
                float v156_data = ir2[2];
                ir2[2] = (v156_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 2))));
                float v159_data = r1[3];
                float v162_data = ir2[3];
                ir2[3] = (v162_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 2))));
                float v165_data = r1[4];
                float v168_data = ir2[4];
                ir2[4] = (v168_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 2))));
                float v171_data = r1[5];
                float v174_data = ir2[5];
                ir2[5] = (v174_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 2))));
                float v177_data = r1[6];
                float v180_data = ir2[6];
                ir2[6] = (v180_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 2))));
                float v183_data = r1[7];
                float v186_data = ir2[7];
                ir2[7] = (v186_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 2))));
              }
              if (v8_lead < 12) {
                float v192_data = r0[3];
                float v193_data = r1[0];
                float v196_data = ir2[0];
                ir2[0] = (v196_data + (v192_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 3))));
                float v199_data = r1[1];
                float v202_data = ir2[1];
                ir2[1] = (v202_data + (v192_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 3))));
                float v205_data = r1[2];
                float v208_data = ir2[2];
                ir2[2] = (v208_data + (v192_data * (sycl::group_broadcast(item.get_sub_group(), v205_data, 3))));
                float v211_data = r1[3];
                float v214_data = ir2[3];
                ir2[3] = (v214_data + (v192_data * (sycl::group_broadcast(item.get_sub_group(), v211_data, 3))));
                float v217_data = r1[4];
                float v220_data = ir2[4];
                ir2[4] = (v220_data + (v192_data * (sycl::group_broadcast(item.get_sub_group(), v217_data, 3))));
                float v223_data = r1[5];
                float v226_data = ir2[5];
                ir2[5] = (v226_data + (v192_data * (sycl::group_broadcast(item.get_sub_group(), v223_data, 3))));
                float v229_data = r1[6];
                float v232_data = ir2[6];
                ir2[6] = (v232_data + (v192_data * (sycl::group_broadcast(item.get_sub_group(), v229_data, 3))));
                float v235_data = r1[7];
                float v238_data = ir2[7];
                ir2[7] = (v238_data + (v192_data * (sycl::group_broadcast(item.get_sub_group(), v235_data, 3))));
              }
              if (v8_lead < 12) {
                float v244_data = r0[4];
                float v245_data = r1[0];
                float v248_data = ir2[0];
                ir2[0] = (v248_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v245_data, 4))));
                float v251_data = r1[1];
                float v254_data = ir2[1];
                ir2[1] = (v254_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v251_data, 4))));
                float v257_data = r1[2];
                float v260_data = ir2[2];
                ir2[2] = (v260_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v257_data, 4))));
                float v263_data = r1[3];
                float v266_data = ir2[3];
                ir2[3] = (v266_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v263_data, 4))));
                float v269_data = r1[4];
                float v272_data = ir2[4];
                ir2[4] = (v272_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v269_data, 4))));
                float v275_data = r1[5];
                float v278_data = ir2[5];
                ir2[5] = (v278_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 4))));
                float v281_data = r1[6];
                float v284_data = ir2[6];
                ir2[6] = (v284_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v281_data, 4))));
                float v287_data = r1[7];
                float v290_data = ir2[7];
                ir2[7] = (v290_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v287_data, 4))));
              }
              if (v8_lead < 12) {
                float v296_data = r0[5];
                float v297_data = r1[0];
                float v300_data = ir2[0];
                ir2[0] = (v300_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v297_data, 5))));
                float v303_data = r1[1];
                float v306_data = ir2[1];
                ir2[1] = (v306_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v303_data, 5))));
                float v309_data = r1[2];
                float v312_data = ir2[2];
                ir2[2] = (v312_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 5))));
                float v315_data = r1[3];
                float v318_data = ir2[3];
                ir2[3] = (v318_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v315_data, 5))));
                float v321_data = r1[4];
                float v324_data = ir2[4];
                ir2[4] = (v324_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v321_data, 5))));
                float v327_data = r1[5];
                float v330_data = ir2[5];
                ir2[5] = (v330_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v327_data, 5))));
                float v333_data = r1[6];
                float v336_data = ir2[6];
                ir2[6] = (v336_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v333_data, 5))));
                float v339_data = r1[7];
                float v342_data = ir2[7];
                ir2[7] = (v342_data + (v296_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 5))));
              }
              if (v8_lead < 12) {
                float v348_data = r0[6];
                float v349_data = r1[0];
                float v352_data = ir2[0];
                ir2[0] = (v352_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v349_data, 6))));
                float v355_data = r1[1];
                float v358_data = ir2[1];
                ir2[1] = (v358_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v355_data, 6))));
                float v361_data = r1[2];
                float v364_data = ir2[2];
                ir2[2] = (v364_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v361_data, 6))));
                float v367_data = r1[3];
                float v370_data = ir2[3];
                ir2[3] = (v370_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v367_data, 6))));
                float v373_data = r1[4];
                float v376_data = ir2[4];
                ir2[4] = (v376_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v373_data, 6))));
                float v379_data = r1[5];
                float v382_data = ir2[5];
                ir2[5] = (v382_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v379_data, 6))));
                float v385_data = r1[6];
                float v388_data = ir2[6];
                ir2[6] = (v388_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v385_data, 6))));
                float v391_data = r1[7];
                float v394_data = ir2[7];
                ir2[7] = (v394_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v391_data, 6))));
              }
              if (v8_lead < 12) {
                float v400_data = r0[7];
                float v401_data = r1[0];
                float v404_data = ir2[0];
                ir2[0] = (v404_data + (v400_data * (sycl::group_broadcast(item.get_sub_group(), v401_data, 7))));
                float v407_data = r1[1];
                float v410_data = ir2[1];
                ir2[1] = (v410_data + (v400_data * (sycl::group_broadcast(item.get_sub_group(), v407_data, 7))));
                float v413_data = r1[2];
                float v416_data = ir2[2];
                ir2[2] = (v416_data + (v400_data * (sycl::group_broadcast(item.get_sub_group(), v413_data, 7))));
                float v419_data = r1[3];
                float v422_data = ir2[3];
                ir2[3] = (v422_data + (v400_data * (sycl::group_broadcast(item.get_sub_group(), v419_data, 7))));
                float v425_data = r1[4];
                float v428_data = ir2[4];
                ir2[4] = (v428_data + (v400_data * (sycl::group_broadcast(item.get_sub_group(), v425_data, 7))));
                float v431_data = r1[5];
                float v434_data = ir2[5];
                ir2[5] = (v434_data + (v400_data * (sycl::group_broadcast(item.get_sub_group(), v431_data, 7))));
                float v437_data = r1[6];
                float v440_data = ir2[6];
                ir2[6] = (v440_data + (v400_data * (sycl::group_broadcast(item.get_sub_group(), v437_data, 7))));
                float v443_data = r1[7];
                float v446_data = ir2[7];
                ir2[7] = (v446_data + (v400_data * (sycl::group_broadcast(item.get_sub_group(), v443_data, 7))));
              }
              if (v8_lead < 12) {
                float v452_data = r0[8];
                float v453_data = r1[0];
                float v456_data = ir2[0];
                ir2[0] = (v456_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v453_data, 8))));
                float v459_data = r1[1];
                float v462_data = ir2[1];
                ir2[1] = (v462_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v459_data, 8))));
                float v465_data = r1[2];
                float v468_data = ir2[2];
                ir2[2] = (v468_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v465_data, 8))));
                float v471_data = r1[3];
                float v474_data = ir2[3];
                ir2[3] = (v474_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v471_data, 8))));
                float v477_data = r1[4];
                float v480_data = ir2[4];
                ir2[4] = (v480_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v477_data, 8))));
                float v483_data = r1[5];
                float v486_data = ir2[5];
                ir2[5] = (v486_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v483_data, 8))));
                float v489_data = r1[6];
                float v492_data = ir2[6];
                ir2[6] = (v492_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v489_data, 8))));
                float v495_data = r1[7];
                float v498_data = ir2[7];
                ir2[7] = (v498_data + (v452_data * (sycl::group_broadcast(item.get_sub_group(), v495_data, 8))));
              }
              if (v8_lead < 12) {
                float v504_data = r0[9];
                float v505_data = r1[0];
                float v508_data = ir2[0];
                ir2[0] = (v508_data + (v504_data * (sycl::group_broadcast(item.get_sub_group(), v505_data, 9))));
                float v511_data = r1[1];
                float v514_data = ir2[1];
                ir2[1] = (v514_data + (v504_data * (sycl::group_broadcast(item.get_sub_group(), v511_data, 9))));
                float v517_data = r1[2];
                float v520_data = ir2[2];
                ir2[2] = (v520_data + (v504_data * (sycl::group_broadcast(item.get_sub_group(), v517_data, 9))));
                float v523_data = r1[3];
                float v526_data = ir2[3];
                ir2[3] = (v526_data + (v504_data * (sycl::group_broadcast(item.get_sub_group(), v523_data, 9))));
                float v529_data = r1[4];
                float v532_data = ir2[4];
                ir2[4] = (v532_data + (v504_data * (sycl::group_broadcast(item.get_sub_group(), v529_data, 9))));
                float v535_data = r1[5];
                float v538_data = ir2[5];
                ir2[5] = (v538_data + (v504_data * (sycl::group_broadcast(item.get_sub_group(), v535_data, 9))));
                float v541_data = r1[6];
                float v544_data = ir2[6];
                ir2[6] = (v544_data + (v504_data * (sycl::group_broadcast(item.get_sub_group(), v541_data, 9))));
                float v547_data = r1[7];
                float v550_data = ir2[7];
                ir2[7] = (v550_data + (v504_data * (sycl::group_broadcast(item.get_sub_group(), v547_data, 9))));
              }
              if (v8_lead < 12) {
                float v556_data = r0[10];
                float v557_data = r1[0];
                float v560_data = ir2[0];
                ir2[0] = (v560_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v557_data, 10))));
                float v563_data = r1[1];
                float v566_data = ir2[1];
                ir2[1] = (v566_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v563_data, 10))));
                float v569_data = r1[2];
                float v572_data = ir2[2];
                ir2[2] = (v572_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v569_data, 10))));
                float v575_data = r1[3];
                float v578_data = ir2[3];
                ir2[3] = (v578_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v575_data, 10))));
                float v581_data = r1[4];
                float v584_data = ir2[4];
                ir2[4] = (v584_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v581_data, 10))));
                float v587_data = r1[5];
                float v590_data = ir2[5];
                ir2[5] = (v590_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v587_data, 10))));
                float v593_data = r1[6];
                float v596_data = ir2[6];
                ir2[6] = (v596_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v593_data, 10))));
                float v599_data = r1[7];
                float v602_data = ir2[7];
                ir2[7] = (v602_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v599_data, 10))));
              }
              if (v8_lead < 12) {
                float v608_data = r0[11];
                float v609_data = r1[0];
                float v612_data = ir2[0];
                ir2[0] = (v612_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v609_data, 11))));
                float v615_data = r1[1];
                float v618_data = ir2[1];
                ir2[1] = (v618_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v615_data, 11))));
                float v621_data = r1[2];
                float v624_data = ir2[2];
                ir2[2] = (v624_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v621_data, 11))));
                float v627_data = r1[3];
                float v630_data = ir2[3];
                ir2[3] = (v630_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v627_data, 11))));
                float v633_data = r1[4];
                float v636_data = ir2[4];
                ir2[4] = (v636_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v633_data, 11))));
                float v639_data = r1[5];
                float v642_data = ir2[5];
                ir2[5] = (v642_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v639_data, 11))));
                float v645_data = r1[6];
                float v648_data = ir2[6];
                ir2[6] = (v648_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v645_data, 11))));
                float v651_data = r1[7];
                float v654_data = ir2[7];
                ir2[7] = (v654_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v651_data, 11))));
              }
              if (v8_lead < 12) {
                float v660_data = r0[12];
                float v661_data = r1[0];
                float v664_data = ir2[0];
                ir2[0] = (v664_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v661_data, 12))));
                float v667_data = r1[1];
                float v670_data = ir2[1];
                ir2[1] = (v670_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v667_data, 12))));
                float v673_data = r1[2];
                float v676_data = ir2[2];
                ir2[2] = (v676_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v673_data, 12))));
                float v679_data = r1[3];
                float v682_data = ir2[3];
                ir2[3] = (v682_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v679_data, 12))));
                float v685_data = r1[4];
                float v688_data = ir2[4];
                ir2[4] = (v688_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v685_data, 12))));
                float v691_data = r1[5];
                float v694_data = ir2[5];
                ir2[5] = (v694_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v691_data, 12))));
                float v697_data = r1[6];
                float v700_data = ir2[6];
                ir2[6] = (v700_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v697_data, 12))));
                float v703_data = r1[7];
                float v706_data = ir2[7];
                ir2[7] = (v706_data + (v660_data * (sycl::group_broadcast(item.get_sub_group(), v703_data, 12))));
              }
              if (v8_lead < 12) {
                float v712_data = r0[13];
                float v713_data = r1[0];
                float v716_data = ir2[0];
                ir2[0] = (v716_data + (v712_data * (sycl::group_broadcast(item.get_sub_group(), v713_data, 13))));
                float v719_data = r1[1];
                float v722_data = ir2[1];
                ir2[1] = (v722_data + (v712_data * (sycl::group_broadcast(item.get_sub_group(), v719_data, 13))));
                float v725_data = r1[2];
                float v728_data = ir2[2];
                ir2[2] = (v728_data + (v712_data * (sycl::group_broadcast(item.get_sub_group(), v725_data, 13))));
                float v731_data = r1[3];
                float v734_data = ir2[3];
                ir2[3] = (v734_data + (v712_data * (sycl::group_broadcast(item.get_sub_group(), v731_data, 13))));
                float v737_data = r1[4];
                float v740_data = ir2[4];
                ir2[4] = (v740_data + (v712_data * (sycl::group_broadcast(item.get_sub_group(), v737_data, 13))));
                float v743_data = r1[5];
                float v746_data = ir2[5];
                ir2[5] = (v746_data + (v712_data * (sycl::group_broadcast(item.get_sub_group(), v743_data, 13))));
                float v749_data = r1[6];
                float v752_data = ir2[6];
                ir2[6] = (v752_data + (v712_data * (sycl::group_broadcast(item.get_sub_group(), v749_data, 13))));
                float v755_data = r1[7];
                float v758_data = ir2[7];
                ir2[7] = (v758_data + (v712_data * (sycl::group_broadcast(item.get_sub_group(), v755_data, 13))));
              }
              if (v8_lead < 12) {
                float v764_data = r0[14];
                float v765_data = r1[0];
                float v768_data = ir2[0];
                ir2[0] = (v768_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v765_data, 14))));
                float v771_data = r1[1];
                float v774_data = ir2[1];
                ir2[1] = (v774_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v771_data, 14))));
                float v777_data = r1[2];
                float v780_data = ir2[2];
                ir2[2] = (v780_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v777_data, 14))));
                float v783_data = r1[3];
                float v786_data = ir2[3];
                ir2[3] = (v786_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v783_data, 14))));
                float v789_data = r1[4];
                float v792_data = ir2[4];
                ir2[4] = (v792_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v789_data, 14))));
                float v795_data = r1[5];
                float v798_data = ir2[5];
                ir2[5] = (v798_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v795_data, 14))));
                float v801_data = r1[6];
                float v804_data = ir2[6];
                ir2[6] = (v804_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v801_data, 14))));
                float v807_data = r1[7];
                float v810_data = ir2[7];
                ir2[7] = (v810_data + (v764_data * (sycl::group_broadcast(item.get_sub_group(), v807_data, 14))));
              }
              if (v8_lead < 12) {
                float v816_data = r0[15];
                float v817_data = r1[0];
                float v820_data = ir2[0];
                ir2[0] = (v820_data + (v816_data * (sycl::group_broadcast(item.get_sub_group(), v817_data, 15))));
                float v823_data = r1[1];
                float v826_data = ir2[1];
                ir2[1] = (v826_data + (v816_data * (sycl::group_broadcast(item.get_sub_group(), v823_data, 15))));
                float v829_data = r1[2];
                float v832_data = ir2[2];
                ir2[2] = (v832_data + (v816_data * (sycl::group_broadcast(item.get_sub_group(), v829_data, 15))));
                float v835_data = r1[3];
                float v838_data = ir2[3];
                ir2[3] = (v838_data + (v816_data * (sycl::group_broadcast(item.get_sub_group(), v835_data, 15))));
                float v841_data = r1[4];
                float v844_data = ir2[4];
                ir2[4] = (v844_data + (v816_data * (sycl::group_broadcast(item.get_sub_group(), v841_data, 15))));
                float v847_data = r1[5];
                float v850_data = ir2[5];
                ir2[5] = (v850_data + (v816_data * (sycl::group_broadcast(item.get_sub_group(), v847_data, 15))));
                float v853_data = r1[6];
                float v856_data = ir2[6];
                ir2[6] = (v856_data + (v816_data * (sycl::group_broadcast(item.get_sub_group(), v853_data, 15))));
                float v859_data = r1[7];
                float v862_data = ir2[7];
                ir2[7] = (v862_data + (v816_data * (sycl::group_broadcast(item.get_sub_group(), v859_data, 15))));
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v868_n1 = 0; v868_n1 < 8; ++v868_n1) {
                  float v870_data = ir2[v868_n1];
                  r2[v868_n1] = v870_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v876_i1 = 0; v876_i1 < 8; ++v876_i1) {
                  float v878_data = r2[v876_i1];
                  glb_m0[(v8_lead + (v876_i1 * 12))] = v878_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

