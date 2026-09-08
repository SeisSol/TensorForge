// === base name ===
kernel_8162b17515

// === header ===
void launcher_kernel_8162b17515(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8162b17515(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8162b17515(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8162b17515(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 24×9(24×9) {0..24}×{0..9} strided
        // m1 24×24(24×24) {0..24}×{0..24} strided
        // m2 24×9(24×9) {0..24}×{0..9} strided
        // m0 24×9(24×9) {0..24}×{0..9} strided({0..24}×{0..9})[0, 1] = m1 24×24(24×24) {0..24}×{0..24} strided({0..24}×{0..24})[0, -1]×m2 24×9(24×9) {0..24}×{0..9} strided({0..24}×{0..9})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 216 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 576 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 216 + 0 + m2_extraOffset];
              float r0[24]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v6_lead = item.get_local_id(0) % 32;
              if (v6_lead < 24) {
                #pragma unroll
                for (int32_t v8_i1 = 0; v8_i1 < 24; ++v8_i1) {
                  float v16_data = glb_m1[(v6_lead + (v8_i1 * 24))];
                  r0[v8_i1] = v16_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              sycl::vec<float, 4> v19_lin = *(sycl::vec<float, 4>*)&glb_m2[0 + item.get_local_id(0) * 4];
              *(sycl::vec<float, 4>*)&r1[0] = v19_lin;
              sycl::vec<float, 2> v20_lin = *(sycl::vec<float, 2>*)&glb_m2[128 + item.get_local_id(0) * 2];
              *(sycl::vec<float, 2>*)&r1[4] = v20_lin;
              float v21_lin = glb_m2[192 + item.get_local_id(0) * 1];
              r1[6] = v21_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // r2 = +(r0 * r1) + None
              // [(0, 24), (0, 9)] [(0, 24)]
              float ir2[9]{};
              if (v6_lead < 24) {
                float v28_data = r0[0];
                float v29_data = r1[0];
                float v32_data = ir2[0];
                ir2[0] = (v32_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 0))));
                float v35_data = r1[1];
                float v38_data = ir2[1];
                ir2[1] = (v38_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 0))));
                float v41_data = r1[2];
                float v44_data = ir2[2];
                ir2[2] = (v44_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 0))));
                float v47_data = r1[3];
                float v50_data = ir2[3];
                ir2[3] = (v50_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 0))));
                float v53_data = r1[4];
                float v56_data = ir2[4];
                ir2[4] = (v56_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 0))));
                float v59_data = r1[5];
                float v62_data = ir2[5];
                ir2[5] = (v62_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 0))));
                float v65_data = r1[6];
                float v68_data = ir2[6];
                ir2[6] = (v68_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 0))));
                float v71_data = r1[7];
                float v74_data = ir2[7];
                ir2[7] = (v74_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 0))));
                float v77_data = r1[8];
                float v80_data = ir2[8];
                ir2[8] = (v80_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 0))));
              }
              if (v6_lead < 24) {
                float v86_data = r0[1];
                float v87_data = r1[0];
                float v90_data = ir2[0];
                ir2[0] = (v90_data + (v86_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 1))));
                float v93_data = r1[1];
                float v96_data = ir2[1];
                ir2[1] = (v96_data + (v86_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 1))));
                float v99_data = r1[2];
                float v102_data = ir2[2];
                ir2[2] = (v102_data + (v86_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 1))));
                float v105_data = r1[3];
                float v108_data = ir2[3];
                ir2[3] = (v108_data + (v86_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 1))));
                float v111_data = r1[4];
                float v114_data = ir2[4];
                ir2[4] = (v114_data + (v86_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 1))));
                float v117_data = r1[5];
                float v120_data = ir2[5];
                ir2[5] = (v120_data + (v86_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 1))));
                float v123_data = r1[6];
                float v126_data = ir2[6];
                ir2[6] = (v126_data + (v86_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 1))));
                float v129_data = r1[7];
                float v132_data = ir2[7];
                ir2[7] = (v132_data + (v86_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 1))));
                float v135_data = r1[8];
                float v138_data = ir2[8];
                ir2[8] = (v138_data + (v86_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 1))));
              }
              if (v6_lead < 24) {
                float v144_data = r0[2];
                float v145_data = r1[0];
                float v148_data = ir2[0];
                ir2[0] = (v148_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 2))));
                float v151_data = r1[1];
                float v154_data = ir2[1];
                ir2[1] = (v154_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 2))));
                float v157_data = r1[2];
                float v160_data = ir2[2];
                ir2[2] = (v160_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 2))));
                float v163_data = r1[3];
                float v166_data = ir2[3];
                ir2[3] = (v166_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 2))));
                float v169_data = r1[4];
                float v172_data = ir2[4];
                ir2[4] = (v172_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 2))));
                float v175_data = r1[5];
                float v178_data = ir2[5];
                ir2[5] = (v178_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 2))));
                float v181_data = r1[6];
                float v184_data = ir2[6];
                ir2[6] = (v184_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 2))));
                float v187_data = r1[7];
                float v190_data = ir2[7];
                ir2[7] = (v190_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 2))));
                float v193_data = r1[8];
                float v196_data = ir2[8];
                ir2[8] = (v196_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 2))));
              }
              if (v6_lead < 24) {
                float v202_data = r0[3];
                float v203_data = r1[0];
                float v206_data = ir2[0];
                ir2[0] = (v206_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 3))));
                float v209_data = r1[1];
                float v212_data = ir2[1];
                ir2[1] = (v212_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v209_data, 3))));
                float v215_data = r1[2];
                float v218_data = ir2[2];
                ir2[2] = (v218_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v215_data, 3))));
                float v221_data = r1[3];
                float v224_data = ir2[3];
                ir2[3] = (v224_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v221_data, 3))));
                float v227_data = r1[4];
                float v230_data = ir2[4];
                ir2[4] = (v230_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v227_data, 3))));
                float v233_data = r1[5];
                float v236_data = ir2[5];
                ir2[5] = (v236_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v233_data, 3))));
                float v239_data = r1[6];
                float v242_data = ir2[6];
                ir2[6] = (v242_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v239_data, 3))));
                float v245_data = r1[7];
                float v248_data = ir2[7];
                ir2[7] = (v248_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v245_data, 3))));
                float v251_data = r1[8];
                float v254_data = ir2[8];
                ir2[8] = (v254_data + (v202_data * (sycl::group_broadcast(item.get_sub_group(), v251_data, 3))));
              }
              if (v6_lead < 24) {
                float v260_data = r0[4];
                float v261_data = r1[0];
                float v264_data = ir2[0];
                ir2[0] = (v264_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v261_data, 4))));
                float v267_data = r1[1];
                float v270_data = ir2[1];
                ir2[1] = (v270_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v267_data, 4))));
                float v273_data = r1[2];
                float v276_data = ir2[2];
                ir2[2] = (v276_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v273_data, 4))));
                float v279_data = r1[3];
                float v282_data = ir2[3];
                ir2[3] = (v282_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 4))));
                float v285_data = r1[4];
                float v288_data = ir2[4];
                ir2[4] = (v288_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v285_data, 4))));
                float v291_data = r1[5];
                float v294_data = ir2[5];
                ir2[5] = (v294_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v291_data, 4))));
                float v297_data = r1[6];
                float v300_data = ir2[6];
                ir2[6] = (v300_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v297_data, 4))));
                float v303_data = r1[7];
                float v306_data = ir2[7];
                ir2[7] = (v306_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v303_data, 4))));
                float v309_data = r1[8];
                float v312_data = ir2[8];
                ir2[8] = (v312_data + (v260_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 4))));
              }
              if (v6_lead < 24) {
                float v318_data = r0[5];
                float v319_data = r1[0];
                float v322_data = ir2[0];
                ir2[0] = (v322_data + (v318_data * (sycl::group_broadcast(item.get_sub_group(), v319_data, 5))));
                float v325_data = r1[1];
                float v328_data = ir2[1];
                ir2[1] = (v328_data + (v318_data * (sycl::group_broadcast(item.get_sub_group(), v325_data, 5))));
                float v331_data = r1[2];
                float v334_data = ir2[2];
                ir2[2] = (v334_data + (v318_data * (sycl::group_broadcast(item.get_sub_group(), v331_data, 5))));
                float v337_data = r1[3];
                float v340_data = ir2[3];
                ir2[3] = (v340_data + (v318_data * (sycl::group_broadcast(item.get_sub_group(), v337_data, 5))));
                float v343_data = r1[4];
                float v346_data = ir2[4];
                ir2[4] = (v346_data + (v318_data * (sycl::group_broadcast(item.get_sub_group(), v343_data, 5))));
                float v349_data = r1[5];
                float v352_data = ir2[5];
                ir2[5] = (v352_data + (v318_data * (sycl::group_broadcast(item.get_sub_group(), v349_data, 5))));
                float v355_data = r1[6];
                float v358_data = ir2[6];
                ir2[6] = (v358_data + (v318_data * (sycl::group_broadcast(item.get_sub_group(), v355_data, 5))));
                float v361_data = r1[7];
                float v364_data = ir2[7];
                ir2[7] = (v364_data + (v318_data * (sycl::group_broadcast(item.get_sub_group(), v361_data, 5))));
                float v367_data = r1[8];
                float v370_data = ir2[8];
                ir2[8] = (v370_data + (v318_data * (sycl::group_broadcast(item.get_sub_group(), v367_data, 5))));
              }
              if (v6_lead < 24) {
                float v376_data = r0[6];
                float v377_data = r1[0];
                float v380_data = ir2[0];
                ir2[0] = (v380_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v377_data, 6))));
                float v383_data = r1[1];
                float v386_data = ir2[1];
                ir2[1] = (v386_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v383_data, 6))));
                float v389_data = r1[2];
                float v392_data = ir2[2];
                ir2[2] = (v392_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v389_data, 6))));
                float v395_data = r1[3];
                float v398_data = ir2[3];
                ir2[3] = (v398_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 6))));
                float v401_data = r1[4];
                float v404_data = ir2[4];
                ir2[4] = (v404_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v401_data, 6))));
                float v407_data = r1[5];
                float v410_data = ir2[5];
                ir2[5] = (v410_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v407_data, 6))));
                float v413_data = r1[6];
                float v416_data = ir2[6];
                ir2[6] = (v416_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v413_data, 6))));
                float v419_data = r1[7];
                float v422_data = ir2[7];
                ir2[7] = (v422_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v419_data, 6))));
                float v425_data = r1[8];
                float v428_data = ir2[8];
                ir2[8] = (v428_data + (v376_data * (sycl::group_broadcast(item.get_sub_group(), v425_data, 6))));
              }
              if (v6_lead < 24) {
                float v434_data = r0[7];
                float v435_data = r1[0];
                float v438_data = ir2[0];
                ir2[0] = (v438_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v435_data, 7))));
                float v441_data = r1[1];
                float v444_data = ir2[1];
                ir2[1] = (v444_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v441_data, 7))));
                float v447_data = r1[2];
                float v450_data = ir2[2];
                ir2[2] = (v450_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v447_data, 7))));
                float v453_data = r1[3];
                float v456_data = ir2[3];
                ir2[3] = (v456_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v453_data, 7))));
                float v459_data = r1[4];
                float v462_data = ir2[4];
                ir2[4] = (v462_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v459_data, 7))));
                float v465_data = r1[5];
                float v468_data = ir2[5];
                ir2[5] = (v468_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v465_data, 7))));
                float v471_data = r1[6];
                float v474_data = ir2[6];
                ir2[6] = (v474_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v471_data, 7))));
                float v477_data = r1[7];
                float v480_data = ir2[7];
                ir2[7] = (v480_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v477_data, 7))));
                float v483_data = r1[8];
                float v486_data = ir2[8];
                ir2[8] = (v486_data + (v434_data * (sycl::group_broadcast(item.get_sub_group(), v483_data, 7))));
              }
              if (v6_lead < 24) {
                float v492_data = r0[8];
                float v493_data = r1[0];
                float v496_data = ir2[0];
                ir2[0] = (v496_data + (v492_data * (sycl::group_broadcast(item.get_sub_group(), v493_data, 8))));
                float v499_data = r1[1];
                float v502_data = ir2[1];
                ir2[1] = (v502_data + (v492_data * (sycl::group_broadcast(item.get_sub_group(), v499_data, 8))));
                float v505_data = r1[2];
                float v508_data = ir2[2];
                ir2[2] = (v508_data + (v492_data * (sycl::group_broadcast(item.get_sub_group(), v505_data, 8))));
                float v511_data = r1[3];
                float v514_data = ir2[3];
                ir2[3] = (v514_data + (v492_data * (sycl::group_broadcast(item.get_sub_group(), v511_data, 8))));
                float v517_data = r1[4];
                float v520_data = ir2[4];
                ir2[4] = (v520_data + (v492_data * (sycl::group_broadcast(item.get_sub_group(), v517_data, 8))));
                float v523_data = r1[5];
                float v526_data = ir2[5];
                ir2[5] = (v526_data + (v492_data * (sycl::group_broadcast(item.get_sub_group(), v523_data, 8))));
                float v529_data = r1[6];
                float v532_data = ir2[6];
                ir2[6] = (v532_data + (v492_data * (sycl::group_broadcast(item.get_sub_group(), v529_data, 8))));
                float v535_data = r1[7];
                float v538_data = ir2[7];
                ir2[7] = (v538_data + (v492_data * (sycl::group_broadcast(item.get_sub_group(), v535_data, 8))));
                float v541_data = r1[8];
                float v544_data = ir2[8];
                ir2[8] = (v544_data + (v492_data * (sycl::group_broadcast(item.get_sub_group(), v541_data, 8))));
              }
              if (v6_lead < 24) {
                float v550_data = r0[9];
                float v551_data = r1[0];
                float v554_data = ir2[0];
                ir2[0] = (v554_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v551_data, 9))));
                float v557_data = r1[1];
                float v560_data = ir2[1];
                ir2[1] = (v560_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v557_data, 9))));
                float v563_data = r1[2];
                float v566_data = ir2[2];
                ir2[2] = (v566_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v563_data, 9))));
                float v569_data = r1[3];
                float v572_data = ir2[3];
                ir2[3] = (v572_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v569_data, 9))));
                float v575_data = r1[4];
                float v578_data = ir2[4];
                ir2[4] = (v578_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v575_data, 9))));
                float v581_data = r1[5];
                float v584_data = ir2[5];
                ir2[5] = (v584_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v581_data, 9))));
                float v587_data = r1[6];
                float v590_data = ir2[6];
                ir2[6] = (v590_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v587_data, 9))));
                float v593_data = r1[7];
                float v596_data = ir2[7];
                ir2[7] = (v596_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v593_data, 9))));
                float v599_data = r1[8];
                float v602_data = ir2[8];
                ir2[8] = (v602_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v599_data, 9))));
              }
              if (v6_lead < 24) {
                float v608_data = r0[10];
                float v609_data = r1[0];
                float v612_data = ir2[0];
                ir2[0] = (v612_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v609_data, 10))));
                float v615_data = r1[1];
                float v618_data = ir2[1];
                ir2[1] = (v618_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v615_data, 10))));
                float v621_data = r1[2];
                float v624_data = ir2[2];
                ir2[2] = (v624_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v621_data, 10))));
                float v627_data = r1[3];
                float v630_data = ir2[3];
                ir2[3] = (v630_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v627_data, 10))));
                float v633_data = r1[4];
                float v636_data = ir2[4];
                ir2[4] = (v636_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v633_data, 10))));
                float v639_data = r1[5];
                float v642_data = ir2[5];
                ir2[5] = (v642_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v639_data, 10))));
                float v645_data = r1[6];
                float v648_data = ir2[6];
                ir2[6] = (v648_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v645_data, 10))));
                float v651_data = r1[7];
                float v654_data = ir2[7];
                ir2[7] = (v654_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v651_data, 10))));
                float v657_data = r1[8];
                float v660_data = ir2[8];
                ir2[8] = (v660_data + (v608_data * (sycl::group_broadcast(item.get_sub_group(), v657_data, 10))));
              }
              if (v6_lead < 24) {
                float v666_data = r0[11];
                float v667_data = r1[0];
                float v670_data = ir2[0];
                ir2[0] = (v670_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v667_data, 11))));
                float v673_data = r1[1];
                float v676_data = ir2[1];
                ir2[1] = (v676_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v673_data, 11))));
                float v679_data = r1[2];
                float v682_data = ir2[2];
                ir2[2] = (v682_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v679_data, 11))));
                float v685_data = r1[3];
                float v688_data = ir2[3];
                ir2[3] = (v688_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v685_data, 11))));
                float v691_data = r1[4];
                float v694_data = ir2[4];
                ir2[4] = (v694_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v691_data, 11))));
                float v697_data = r1[5];
                float v700_data = ir2[5];
                ir2[5] = (v700_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v697_data, 11))));
                float v703_data = r1[6];
                float v706_data = ir2[6];
                ir2[6] = (v706_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v703_data, 11))));
                float v709_data = r1[7];
                float v712_data = ir2[7];
                ir2[7] = (v712_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v709_data, 11))));
                float v715_data = r1[8];
                float v718_data = ir2[8];
                ir2[8] = (v718_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v715_data, 11))));
              }
              if (v6_lead < 24) {
                float v724_data = r0[12];
                float v725_data = r1[0];
                float v728_data = ir2[0];
                ir2[0] = (v728_data + (v724_data * (sycl::group_broadcast(item.get_sub_group(), v725_data, 12))));
                float v731_data = r1[1];
                float v734_data = ir2[1];
                ir2[1] = (v734_data + (v724_data * (sycl::group_broadcast(item.get_sub_group(), v731_data, 12))));
                float v737_data = r1[2];
                float v740_data = ir2[2];
                ir2[2] = (v740_data + (v724_data * (sycl::group_broadcast(item.get_sub_group(), v737_data, 12))));
                float v743_data = r1[3];
                float v746_data = ir2[3];
                ir2[3] = (v746_data + (v724_data * (sycl::group_broadcast(item.get_sub_group(), v743_data, 12))));
                float v749_data = r1[4];
                float v752_data = ir2[4];
                ir2[4] = (v752_data + (v724_data * (sycl::group_broadcast(item.get_sub_group(), v749_data, 12))));
                float v755_data = r1[5];
                float v758_data = ir2[5];
                ir2[5] = (v758_data + (v724_data * (sycl::group_broadcast(item.get_sub_group(), v755_data, 12))));
                float v761_data = r1[6];
                float v764_data = ir2[6];
                ir2[6] = (v764_data + (v724_data * (sycl::group_broadcast(item.get_sub_group(), v761_data, 12))));
                float v767_data = r1[7];
                float v770_data = ir2[7];
                ir2[7] = (v770_data + (v724_data * (sycl::group_broadcast(item.get_sub_group(), v767_data, 12))));
                float v773_data = r1[8];
                float v776_data = ir2[8];
                ir2[8] = (v776_data + (v724_data * (sycl::group_broadcast(item.get_sub_group(), v773_data, 12))));
              }
              if (v6_lead < 24) {
                float v782_data = r0[13];
                float v783_data = r1[0];
                float v786_data = ir2[0];
                ir2[0] = (v786_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v783_data, 13))));
                float v789_data = r1[1];
                float v792_data = ir2[1];
                ir2[1] = (v792_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v789_data, 13))));
                float v795_data = r1[2];
                float v798_data = ir2[2];
                ir2[2] = (v798_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v795_data, 13))));
                float v801_data = r1[3];
                float v804_data = ir2[3];
                ir2[3] = (v804_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v801_data, 13))));
                float v807_data = r1[4];
                float v810_data = ir2[4];
                ir2[4] = (v810_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v807_data, 13))));
                float v813_data = r1[5];
                float v816_data = ir2[5];
                ir2[5] = (v816_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v813_data, 13))));
                float v819_data = r1[6];
                float v822_data = ir2[6];
                ir2[6] = (v822_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v819_data, 13))));
                float v825_data = r1[7];
                float v828_data = ir2[7];
                ir2[7] = (v828_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v825_data, 13))));
                float v831_data = r1[8];
                float v834_data = ir2[8];
                ir2[8] = (v834_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v831_data, 13))));
              }
              if (v6_lead < 24) {
                float v840_data = r0[14];
                float v841_data = r1[0];
                float v844_data = ir2[0];
                ir2[0] = (v844_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v841_data, 14))));
                float v847_data = r1[1];
                float v850_data = ir2[1];
                ir2[1] = (v850_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v847_data, 14))));
                float v853_data = r1[2];
                float v856_data = ir2[2];
                ir2[2] = (v856_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v853_data, 14))));
                float v859_data = r1[3];
                float v862_data = ir2[3];
                ir2[3] = (v862_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v859_data, 14))));
                float v865_data = r1[4];
                float v868_data = ir2[4];
                ir2[4] = (v868_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v865_data, 14))));
                float v871_data = r1[5];
                float v874_data = ir2[5];
                ir2[5] = (v874_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v871_data, 14))));
                float v877_data = r1[6];
                float v880_data = ir2[6];
                ir2[6] = (v880_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v877_data, 14))));
                float v883_data = r1[7];
                float v886_data = ir2[7];
                ir2[7] = (v886_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v883_data, 14))));
                float v889_data = r1[8];
                float v892_data = ir2[8];
                ir2[8] = (v892_data + (v840_data * (sycl::group_broadcast(item.get_sub_group(), v889_data, 14))));
              }
              if (v6_lead < 24) {
                float v898_data = r0[15];
                float v899_data = r1[0];
                float v902_data = ir2[0];
                ir2[0] = (v902_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v899_data, 15))));
                float v905_data = r1[1];
                float v908_data = ir2[1];
                ir2[1] = (v908_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v905_data, 15))));
                float v911_data = r1[2];
                float v914_data = ir2[2];
                ir2[2] = (v914_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v911_data, 15))));
                float v917_data = r1[3];
                float v920_data = ir2[3];
                ir2[3] = (v920_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v917_data, 15))));
                float v923_data = r1[4];
                float v926_data = ir2[4];
                ir2[4] = (v926_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v923_data, 15))));
                float v929_data = r1[5];
                float v932_data = ir2[5];
                ir2[5] = (v932_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v929_data, 15))));
                float v935_data = r1[6];
                float v938_data = ir2[6];
                ir2[6] = (v938_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v935_data, 15))));
                float v941_data = r1[7];
                float v944_data = ir2[7];
                ir2[7] = (v944_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v941_data, 15))));
                float v947_data = r1[8];
                float v950_data = ir2[8];
                ir2[8] = (v950_data + (v898_data * (sycl::group_broadcast(item.get_sub_group(), v947_data, 15))));
              }
              if (v6_lead < 24) {
                float v956_data = r0[16];
                float v957_data = r1[0];
                float v960_data = ir2[0];
                ir2[0] = (v960_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v957_data, 16))));
                float v963_data = r1[1];
                float v966_data = ir2[1];
                ir2[1] = (v966_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v963_data, 16))));
                float v969_data = r1[2];
                float v972_data = ir2[2];
                ir2[2] = (v972_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v969_data, 16))));
                float v975_data = r1[3];
                float v978_data = ir2[3];
                ir2[3] = (v978_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v975_data, 16))));
                float v981_data = r1[4];
                float v984_data = ir2[4];
                ir2[4] = (v984_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v981_data, 16))));
                float v987_data = r1[5];
                float v990_data = ir2[5];
                ir2[5] = (v990_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v987_data, 16))));
                float v993_data = r1[6];
                float v996_data = ir2[6];
                ir2[6] = (v996_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v993_data, 16))));
                float v999_data = r1[7];
                float v1002_data = ir2[7];
                ir2[7] = (v1002_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v999_data, 16))));
                float v1005_data = r1[8];
                float v1008_data = ir2[8];
                ir2[8] = (v1008_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v1005_data, 16))));
              }
              if (v6_lead < 24) {
                float v1014_data = r0[17];
                float v1015_data = r1[0];
                float v1018_data = ir2[0];
                ir2[0] = (v1018_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v1015_data, 17))));
                float v1021_data = r1[1];
                float v1024_data = ir2[1];
                ir2[1] = (v1024_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v1021_data, 17))));
                float v1027_data = r1[2];
                float v1030_data = ir2[2];
                ir2[2] = (v1030_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v1027_data, 17))));
                float v1033_data = r1[3];
                float v1036_data = ir2[3];
                ir2[3] = (v1036_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v1033_data, 17))));
                float v1039_data = r1[4];
                float v1042_data = ir2[4];
                ir2[4] = (v1042_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v1039_data, 17))));
                float v1045_data = r1[5];
                float v1048_data = ir2[5];
                ir2[5] = (v1048_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v1045_data, 17))));
                float v1051_data = r1[6];
                float v1054_data = ir2[6];
                ir2[6] = (v1054_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v1051_data, 17))));
                float v1057_data = r1[7];
                float v1060_data = ir2[7];
                ir2[7] = (v1060_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v1057_data, 17))));
                float v1063_data = r1[8];
                float v1066_data = ir2[8];
                ir2[8] = (v1066_data + (v1014_data * (sycl::group_broadcast(item.get_sub_group(), v1063_data, 17))));
              }
              if (v6_lead < 24) {
                float v1072_data = r0[18];
                float v1073_data = r1[0];
                float v1076_data = ir2[0];
                ir2[0] = (v1076_data + (v1072_data * (sycl::group_broadcast(item.get_sub_group(), v1073_data, 18))));
                float v1079_data = r1[1];
                float v1082_data = ir2[1];
                ir2[1] = (v1082_data + (v1072_data * (sycl::group_broadcast(item.get_sub_group(), v1079_data, 18))));
                float v1085_data = r1[2];
                float v1088_data = ir2[2];
                ir2[2] = (v1088_data + (v1072_data * (sycl::group_broadcast(item.get_sub_group(), v1085_data, 18))));
                float v1091_data = r1[3];
                float v1094_data = ir2[3];
                ir2[3] = (v1094_data + (v1072_data * (sycl::group_broadcast(item.get_sub_group(), v1091_data, 18))));
                float v1097_data = r1[4];
                float v1100_data = ir2[4];
                ir2[4] = (v1100_data + (v1072_data * (sycl::group_broadcast(item.get_sub_group(), v1097_data, 18))));
                float v1103_data = r1[5];
                float v1106_data = ir2[5];
                ir2[5] = (v1106_data + (v1072_data * (sycl::group_broadcast(item.get_sub_group(), v1103_data, 18))));
                float v1109_data = r1[6];
                float v1112_data = ir2[6];
                ir2[6] = (v1112_data + (v1072_data * (sycl::group_broadcast(item.get_sub_group(), v1109_data, 18))));
                float v1115_data = r1[7];
                float v1118_data = ir2[7];
                ir2[7] = (v1118_data + (v1072_data * (sycl::group_broadcast(item.get_sub_group(), v1115_data, 18))));
                float v1121_data = r1[8];
                float v1124_data = ir2[8];
                ir2[8] = (v1124_data + (v1072_data * (sycl::group_broadcast(item.get_sub_group(), v1121_data, 18))));
              }
              if (v6_lead < 24) {
                float v1130_data = r0[19];
                float v1131_data = r1[0];
                float v1134_data = ir2[0];
                ir2[0] = (v1134_data + (v1130_data * (sycl::group_broadcast(item.get_sub_group(), v1131_data, 19))));
                float v1137_data = r1[1];
                float v1140_data = ir2[1];
                ir2[1] = (v1140_data + (v1130_data * (sycl::group_broadcast(item.get_sub_group(), v1137_data, 19))));
                float v1143_data = r1[2];
                float v1146_data = ir2[2];
                ir2[2] = (v1146_data + (v1130_data * (sycl::group_broadcast(item.get_sub_group(), v1143_data, 19))));
                float v1149_data = r1[3];
                float v1152_data = ir2[3];
                ir2[3] = (v1152_data + (v1130_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 19))));
                float v1155_data = r1[4];
                float v1158_data = ir2[4];
                ir2[4] = (v1158_data + (v1130_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 19))));
                float v1161_data = r1[5];
                float v1164_data = ir2[5];
                ir2[5] = (v1164_data + (v1130_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 19))));
                float v1167_data = r1[6];
                float v1170_data = ir2[6];
                ir2[6] = (v1170_data + (v1130_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 19))));
                float v1173_data = r1[7];
                float v1176_data = ir2[7];
                ir2[7] = (v1176_data + (v1130_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 19))));
                float v1179_data = r1[8];
                float v1182_data = ir2[8];
                ir2[8] = (v1182_data + (v1130_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 19))));
              }
              if (v6_lead < 24) {
                float v1188_data = r0[20];
                float v1189_data = r1[0];
                float v1192_data = ir2[0];
                ir2[0] = (v1192_data + (v1188_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 20))));
                float v1195_data = r1[1];
                float v1198_data = ir2[1];
                ir2[1] = (v1198_data + (v1188_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 20))));
                float v1201_data = r1[2];
                float v1204_data = ir2[2];
                ir2[2] = (v1204_data + (v1188_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 20))));
                float v1207_data = r1[3];
                float v1210_data = ir2[3];
                ir2[3] = (v1210_data + (v1188_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 20))));
                float v1213_data = r1[4];
                float v1216_data = ir2[4];
                ir2[4] = (v1216_data + (v1188_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 20))));
                float v1219_data = r1[5];
                float v1222_data = ir2[5];
                ir2[5] = (v1222_data + (v1188_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 20))));
                float v1225_data = r1[6];
                float v1228_data = ir2[6];
                ir2[6] = (v1228_data + (v1188_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 20))));
                float v1231_data = r1[7];
                float v1234_data = ir2[7];
                ir2[7] = (v1234_data + (v1188_data * (sycl::group_broadcast(item.get_sub_group(), v1231_data, 20))));
                float v1237_data = r1[8];
                float v1240_data = ir2[8];
                ir2[8] = (v1240_data + (v1188_data * (sycl::group_broadcast(item.get_sub_group(), v1237_data, 20))));
              }
              if (v6_lead < 24) {
                float v1246_data = r0[21];
                float v1247_data = r1[0];
                float v1250_data = ir2[0];
                ir2[0] = (v1250_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1247_data, 21))));
                float v1253_data = r1[1];
                float v1256_data = ir2[1];
                ir2[1] = (v1256_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1253_data, 21))));
                float v1259_data = r1[2];
                float v1262_data = ir2[2];
                ir2[2] = (v1262_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1259_data, 21))));
                float v1265_data = r1[3];
                float v1268_data = ir2[3];
                ir2[3] = (v1268_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1265_data, 21))));
                float v1271_data = r1[4];
                float v1274_data = ir2[4];
                ir2[4] = (v1274_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1271_data, 21))));
                float v1277_data = r1[5];
                float v1280_data = ir2[5];
                ir2[5] = (v1280_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1277_data, 21))));
                float v1283_data = r1[6];
                float v1286_data = ir2[6];
                ir2[6] = (v1286_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1283_data, 21))));
                float v1289_data = r1[7];
                float v1292_data = ir2[7];
                ir2[7] = (v1292_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1289_data, 21))));
                float v1295_data = r1[8];
                float v1298_data = ir2[8];
                ir2[8] = (v1298_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1295_data, 21))));
              }
              if (v6_lead < 24) {
                float v1304_data = r0[22];
                float v1305_data = r1[0];
                float v1308_data = ir2[0];
                ir2[0] = (v1308_data + (v1304_data * (sycl::group_broadcast(item.get_sub_group(), v1305_data, 22))));
                float v1311_data = r1[1];
                float v1314_data = ir2[1];
                ir2[1] = (v1314_data + (v1304_data * (sycl::group_broadcast(item.get_sub_group(), v1311_data, 22))));
                float v1317_data = r1[2];
                float v1320_data = ir2[2];
                ir2[2] = (v1320_data + (v1304_data * (sycl::group_broadcast(item.get_sub_group(), v1317_data, 22))));
                float v1323_data = r1[3];
                float v1326_data = ir2[3];
                ir2[3] = (v1326_data + (v1304_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 22))));
                float v1329_data = r1[4];
                float v1332_data = ir2[4];
                ir2[4] = (v1332_data + (v1304_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 22))));
                float v1335_data = r1[5];
                float v1338_data = ir2[5];
                ir2[5] = (v1338_data + (v1304_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 22))));
                float v1341_data = r1[6];
                float v1344_data = ir2[6];
                ir2[6] = (v1344_data + (v1304_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 22))));
                float v1347_data = r1[7];
                float v1350_data = ir2[7];
                ir2[7] = (v1350_data + (v1304_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 22))));
                float v1353_data = r1[8];
                float v1356_data = ir2[8];
                ir2[8] = (v1356_data + (v1304_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 22))));
              }
              if (v6_lead < 24) {
                float v1362_data = r0[23];
                float v1363_data = r1[0];
                float v1366_data = ir2[0];
                ir2[0] = (v1366_data + (v1362_data * (sycl::group_broadcast(item.get_sub_group(), v1363_data, 23))));
                float v1369_data = r1[1];
                float v1372_data = ir2[1];
                ir2[1] = (v1372_data + (v1362_data * (sycl::group_broadcast(item.get_sub_group(), v1369_data, 23))));
                float v1375_data = r1[2];
                float v1378_data = ir2[2];
                ir2[2] = (v1378_data + (v1362_data * (sycl::group_broadcast(item.get_sub_group(), v1375_data, 23))));
                float v1381_data = r1[3];
                float v1384_data = ir2[3];
                ir2[3] = (v1384_data + (v1362_data * (sycl::group_broadcast(item.get_sub_group(), v1381_data, 23))));
                float v1387_data = r1[4];
                float v1390_data = ir2[4];
                ir2[4] = (v1390_data + (v1362_data * (sycl::group_broadcast(item.get_sub_group(), v1387_data, 23))));
                float v1393_data = r1[5];
                float v1396_data = ir2[5];
                ir2[5] = (v1396_data + (v1362_data * (sycl::group_broadcast(item.get_sub_group(), v1393_data, 23))));
                float v1399_data = r1[6];
                float v1402_data = ir2[6];
                ir2[6] = (v1402_data + (v1362_data * (sycl::group_broadcast(item.get_sub_group(), v1399_data, 23))));
                float v1405_data = r1[7];
                float v1408_data = ir2[7];
                ir2[7] = (v1408_data + (v1362_data * (sycl::group_broadcast(item.get_sub_group(), v1405_data, 23))));
                float v1411_data = r1[8];
                float v1414_data = ir2[8];
                ir2[8] = (v1414_data + (v1362_data * (sycl::group_broadcast(item.get_sub_group(), v1411_data, 23))));
              }
              if (v6_lead < 24) {
                #pragma unroll
                for (int32_t v1420_n1 = 0; v1420_n1 < 9; ++v1420_n1) {
                  float v1422_data = ir2[v1420_n1];
                  r2[v1420_n1] = v1422_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v6_lead < 24) {
                #pragma unroll
                for (int32_t v1428_i1 = 0; v1428_i1 < 9; ++v1428_i1) {
                  float v1430_data = r2[v1428_i1];
                  glb_m0[(v6_lead + (v1428_i1 * 24))] = v1430_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

