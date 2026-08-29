// === base name ===
kernel_08703cce1d

// === header ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08703cce1d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08703cce1d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32×32(12×6) {0..12}×{0..6} strided
        // m1 32×32(6×6) {0..6}×{0..6} strided
        // m2 32×32(12×6) {0..12}×{0..6} strided
        // m3 32×32(12×12) {0..12}×{0..12} strided
        // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
        // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
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
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              float r0[6]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v9_lead = item.get_local_id(0) % 16;
              if (v9_lead < 12) {
                #pragma unroll
                for (int32_t v11_i1 = 0; v11_i1 < 6; ++v11_i1) {
                  float v19_data = glb_m0[(v9_lead + (v11_i1 * 12))];
                  r0[v11_i1] = v19_data;
                }
              }
              float r1[6]{};
              // r1 = load{g>r}(glb_m1);
              float v22_lin = glb_m1[0 + item.get_local_id(0) * 1];
              r1[0] = v22_lin;
              float v23_lin = glb_m1[16 + item.get_local_id(0) * 1];
              r1[1] = v23_lin;
              float v24_lin = glb_m1[32 + item.get_local_id(0) * 1];
              r1[2] = v24_lin;
              float v25_lin = glb_m1[48 + item.get_local_id(0) * 1];
              r1[3] = v25_lin;
              float v26_lin = glb_m1[64 + item.get_local_id(0) * 1];
              r1[4] = v26_lin;
              float v27_lin = glb_m1[80 + item.get_local_id(0) * 1];
              r1[5] = v27_lin;
              float v28_lin = glb_m1[96 + item.get_local_id(0) * 1];
              r1[6] = v28_lin;
              float v29_lin = glb_m1[112 + item.get_local_id(0) * 1];
              r1[7] = v29_lin;
              float v30_lin = glb_m1[128 + item.get_local_id(0) * 1];
              r1[8] = v30_lin;
              float v31_lin = glb_m1[144 + item.get_local_id(0) * 1];
              r1[9] = v31_lin;
              float v32_lin = glb_m1[160 + item.get_local_id(0) * 1];
              r1[10] = v32_lin;
              float v33_lin = glb_m1[176 + item.get_local_id(0) * 1];
              r1[11] = v33_lin;
              float v34_lin = glb_m1[192 + item.get_local_id(0) * 1];
              r1[12] = v34_lin;
              float v35_lin = glb_m1[208 + item.get_local_id(0) * 1];
              r1[13] = v35_lin;
              float v36_lin = glb_m1[224 + item.get_local_id(0) * 1];
              r1[14] = v36_lin;
              float v37_lin = glb_m1[240 + item.get_local_id(0) * 1];
              r1[15] = v37_lin;
              float v38_lin = glb_m1[256 + item.get_local_id(0) * 1];
              r1[16] = v38_lin;
              float v39_lin = glb_m1[272 + item.get_local_id(0) * 1];
              r1[17] = v39_lin;
              float v40_lin = glb_m1[288 + item.get_local_id(0) * 1];
              r1[18] = v40_lin;
              float v41_lin = glb_m1[304 + item.get_local_id(0) * 1];
              r1[19] = v41_lin;
              float v42_lin = glb_m1[320 + item.get_local_id(0) * 1];
              r1[20] = v42_lin;
              float v43_lin = glb_m1[336 + item.get_local_id(0) * 1];
              r1[21] = v43_lin;
              float v44_lin = glb_m1[352 + item.get_local_id(0) * 1];
              r1[22] = v44_lin;
              float v45_lin = glb_m1[368 + item.get_local_id(0) * 1];
              r1[23] = v45_lin;
              float v46_lin = glb_m1[384 + item.get_local_id(0) * 1];
              r1[24] = v46_lin;
              float v47_lin = glb_m1[400 + item.get_local_id(0) * 1];
              r1[25] = v47_lin;
              float v48_lin = glb_m1[416 + item.get_local_id(0) * 1];
              r1[26] = v48_lin;
              float v49_lin = glb_m1[432 + item.get_local_id(0) * 1];
              r1[27] = v49_lin;
              float v50_lin = glb_m1[448 + item.get_local_id(0) * 1];
              r1[28] = v50_lin;
              float v51_lin = glb_m1[464 + item.get_local_id(0) * 1];
              r1[29] = v51_lin;
              float v52_lin = glb_m1[480 + item.get_local_id(0) * 1];
              r1[30] = v52_lin;
              float v53_lin = glb_m1[496 + item.get_local_id(0) * 1];
              r1[31] = v53_lin;
              float v54_lin = glb_m1[512 + item.get_local_id(0) * 1];
              r1[32] = v54_lin;
              float v55_lin = glb_m1[528 + item.get_local_id(0) * 1];
              r1[33] = v55_lin;
              float v56_lin = glb_m1[544 + item.get_local_id(0) * 1];
              r1[34] = v56_lin;
              float v57_lin = glb_m1[560 + item.get_local_id(0) * 1];
              r1[35] = v57_lin;
              float v58_lin = glb_m1[576 + item.get_local_id(0) * 1];
              r1[36] = v58_lin;
              float v59_lin = glb_m1[592 + item.get_local_id(0) * 1];
              r1[37] = v59_lin;
              float v60_lin = glb_m1[608 + item.get_local_id(0) * 1];
              r1[38] = v60_lin;
              float v61_lin = glb_m1[624 + item.get_local_id(0) * 1];
              r1[39] = v61_lin;
              float v62_lin = glb_m1[640 + item.get_local_id(0) * 1];
              r1[40] = v62_lin;
              float v63_lin = glb_m1[656 + item.get_local_id(0) * 1];
              r1[41] = v63_lin;
              float v64_lin = glb_m1[672 + item.get_local_id(0) * 1];
              r1[42] = v64_lin;
              float v65_lin = glb_m1[688 + item.get_local_id(0) * 1];
              r1[43] = v65_lin;
              float v66_lin = glb_m1[704 + item.get_local_id(0) * 1];
              r1[44] = v66_lin;
              float v67_lin = glb_m1[720 + item.get_local_id(0) * 1];
              r1[45] = v67_lin;
              float v68_lin = glb_m1[736 + item.get_local_id(0) * 1];
              r1[46] = v68_lin;
              float v69_lin = glb_m1[752 + item.get_local_id(0) * 1];
              r1[47] = v69_lin;
              float v70_lin = glb_m1[768 + item.get_local_id(0) * 1];
              r1[48] = v70_lin;
              float v71_lin = glb_m1[784 + item.get_local_id(0) * 1];
              r1[49] = v71_lin;
              float v72_lin = glb_m1[800 + item.get_local_id(0) * 1];
              r1[50] = v72_lin;
              float v73_lin = glb_m1[816 + item.get_local_id(0) * 1];
              r1[51] = v73_lin;
              float v74_lin = glb_m1[832 + item.get_local_id(0) * 1];
              r1[52] = v74_lin;
              float v75_lin = glb_m1[848 + item.get_local_id(0) * 1];
              r1[53] = v75_lin;
              float v76_lin = glb_m1[864 + item.get_local_id(0) * 1];
              r1[54] = v76_lin;
              float v77_lin = glb_m1[880 + item.get_local_id(0) * 1];
              r1[55] = v77_lin;
              float v78_lin = glb_m1[896 + item.get_local_id(0) * 1];
              r1[56] = v78_lin;
              float v79_lin = glb_m1[912 + item.get_local_id(0) * 1];
              r1[57] = v79_lin;
              float v80_lin = glb_m1[928 + item.get_local_id(0) * 1];
              r1[58] = v80_lin;
              float v81_lin = glb_m1[944 + item.get_local_id(0) * 1];
              r1[59] = v81_lin;
              float v82_lin = glb_m1[960 + item.get_local_id(0) * 1];
              r1[60] = v82_lin;
              float v83_lin = glb_m1[976 + item.get_local_id(0) * 1];
              r1[61] = v83_lin;
              float v84_lin = glb_m1[992 + item.get_local_id(0) * 1];
              r1[62] = v84_lin;
              float v85_lin = glb_m1[1008 + item.get_local_id(0) * 1];
              r1[63] = v85_lin;
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v9_lead < 12) {
                #pragma unroll
                for (int32_t v91_i1 = 0; v91_i1 < 12; ++v91_i1) {
                  float v99_data = glb_m3[(v9_lead + (v91_i1 * 12))];
                  r3[v91_i1] = v99_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[6]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              if (v9_lead < 12) {
                float v106_data = r0[0];
                float v107_data = r1[0];
                float v110_data = r2[0];
                r2[0] = (v110_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 0))));
                float v113_data = r1[1];
                float v116_data = r2[1];
                r2[1] = (v116_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 0))));
                float v119_data = r1[2];
                float v122_data = r2[2];
                r2[2] = (v122_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 0))));
                float v125_data = r1[3];
                float v128_data = r2[3];
                r2[3] = (v128_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 0))));
                float v131_data = r1[4];
                float v134_data = r2[4];
                r2[4] = (v134_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 0))));
                float v137_data = r1[5];
                float v140_data = r2[5];
                r2[5] = (v140_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 0))));
              }
              if (v9_lead < 12) {
                float v146_data = r0[1];
                float v147_data = r1[0];
                float v150_data = r2[0];
                r2[0] = (v150_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 1))));
                float v153_data = r1[1];
                float v156_data = r2[1];
                r2[1] = (v156_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 1))));
                float v159_data = r1[2];
                float v162_data = r2[2];
                r2[2] = (v162_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 1))));
                float v165_data = r1[3];
                float v168_data = r2[3];
                r2[3] = (v168_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v165_data, 1))));
                float v171_data = r1[4];
                float v174_data = r2[4];
                r2[4] = (v174_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 1))));
                float v177_data = r1[5];
                float v180_data = r2[5];
                r2[5] = (v180_data + (v146_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 1))));
              }
              if (v9_lead < 12) {
                float v186_data = r0[2];
                float v187_data = r1[0];
                float v190_data = r2[0];
                r2[0] = (v190_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 2))));
                float v193_data = r1[1];
                float v196_data = r2[1];
                r2[1] = (v196_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 2))));
                float v199_data = r1[2];
                float v202_data = r2[2];
                r2[2] = (v202_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 2))));
                float v205_data = r1[3];
                float v208_data = r2[3];
                r2[3] = (v208_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v205_data, 2))));
                float v211_data = r1[4];
                float v214_data = r2[4];
                r2[4] = (v214_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v211_data, 2))));
                float v217_data = r1[5];
                float v220_data = r2[5];
                r2[5] = (v220_data + (v186_data * (sycl::group_broadcast(item.get_sub_group(), v217_data, 2))));
              }
              if (v9_lead < 12) {
                float v226_data = r0[3];
                float v227_data = r1[0];
                float v230_data = r2[0];
                r2[0] = (v230_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v227_data, 3))));
                float v233_data = r1[1];
                float v236_data = r2[1];
                r2[1] = (v236_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v233_data, 3))));
                float v239_data = r1[2];
                float v242_data = r2[2];
                r2[2] = (v242_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v239_data, 3))));
                float v245_data = r1[3];
                float v248_data = r2[3];
                r2[3] = (v248_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v245_data, 3))));
                float v251_data = r1[4];
                float v254_data = r2[4];
                r2[4] = (v254_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v251_data, 3))));
                float v257_data = r1[5];
                float v260_data = r2[5];
                r2[5] = (v260_data + (v226_data * (sycl::group_broadcast(item.get_sub_group(), v257_data, 3))));
              }
              if (v9_lead < 12) {
                float v266_data = r0[4];
                float v267_data = r1[0];
                float v270_data = r2[0];
                r2[0] = (v270_data + (v266_data * (sycl::group_broadcast(item.get_sub_group(), v267_data, 4))));
                float v273_data = r1[1];
                float v276_data = r2[1];
                r2[1] = (v276_data + (v266_data * (sycl::group_broadcast(item.get_sub_group(), v273_data, 4))));
                float v279_data = r1[2];
                float v282_data = r2[2];
                r2[2] = (v282_data + (v266_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 4))));
                float v285_data = r1[3];
                float v288_data = r2[3];
                r2[3] = (v288_data + (v266_data * (sycl::group_broadcast(item.get_sub_group(), v285_data, 4))));
                float v291_data = r1[4];
                float v294_data = r2[4];
                r2[4] = (v294_data + (v266_data * (sycl::group_broadcast(item.get_sub_group(), v291_data, 4))));
                float v297_data = r1[5];
                float v300_data = r2[5];
                r2[5] = (v300_data + (v266_data * (sycl::group_broadcast(item.get_sub_group(), v297_data, 4))));
              }
              if (v9_lead < 12) {
                float v306_data = r0[5];
                float v307_data = r1[0];
                float v310_data = r2[0];
                r2[0] = (v310_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v307_data, 5))));
                float v313_data = r1[1];
                float v316_data = r2[1];
                r2[1] = (v316_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v313_data, 5))));
                float v319_data = r1[2];
                float v322_data = r2[2];
                r2[2] = (v322_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v319_data, 5))));
                float v325_data = r1[3];
                float v328_data = r2[3];
                r2[3] = (v328_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v325_data, 5))));
                float v331_data = r1[4];
                float v334_data = r2[4];
                r2[4] = (v334_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v331_data, 5))));
                float v337_data = r1[5];
                float v340_data = r2[5];
                r2[5] = (v340_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v337_data, 5))));
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r4[6]{};
              // r4 = +(r3 * r2) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir4[6]{};
              if (v9_lead < 12) {
                float v348_data = r3[0];
                float v349_data = r2[0];
                float v352_data = ir4[0];
                ir4[0] = (v352_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v349_data, 0))));
                float v355_data = r2[1];
                float v358_data = ir4[1];
                ir4[1] = (v358_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v355_data, 0))));
                float v361_data = r2[2];
                float v364_data = ir4[2];
                ir4[2] = (v364_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v361_data, 0))));
                float v367_data = r2[3];
                float v370_data = ir4[3];
                ir4[3] = (v370_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v367_data, 0))));
                float v373_data = r2[4];
                float v376_data = ir4[4];
                ir4[4] = (v376_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v373_data, 0))));
                float v379_data = r2[5];
                float v382_data = ir4[5];
                ir4[5] = (v382_data + (v348_data * (sycl::group_broadcast(item.get_sub_group(), v379_data, 0))));
              }
              if (v9_lead < 12) {
                float v388_data = r3[1];
                float v389_data = r2[0];
                float v392_data = ir4[0];
                ir4[0] = (v392_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v389_data, 1))));
                float v395_data = r2[1];
                float v398_data = ir4[1];
                ir4[1] = (v398_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 1))));
                float v401_data = r2[2];
                float v404_data = ir4[2];
                ir4[2] = (v404_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v401_data, 1))));
                float v407_data = r2[3];
                float v410_data = ir4[3];
                ir4[3] = (v410_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v407_data, 1))));
                float v413_data = r2[4];
                float v416_data = ir4[4];
                ir4[4] = (v416_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v413_data, 1))));
                float v419_data = r2[5];
                float v422_data = ir4[5];
                ir4[5] = (v422_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v419_data, 1))));
              }
              if (v9_lead < 12) {
                float v428_data = r3[2];
                float v429_data = r2[0];
                float v432_data = ir4[0];
                ir4[0] = (v432_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v429_data, 2))));
                float v435_data = r2[1];
                float v438_data = ir4[1];
                ir4[1] = (v438_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v435_data, 2))));
                float v441_data = r2[2];
                float v444_data = ir4[2];
                ir4[2] = (v444_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v441_data, 2))));
                float v447_data = r2[3];
                float v450_data = ir4[3];
                ir4[3] = (v450_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v447_data, 2))));
                float v453_data = r2[4];
                float v456_data = ir4[4];
                ir4[4] = (v456_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v453_data, 2))));
                float v459_data = r2[5];
                float v462_data = ir4[5];
                ir4[5] = (v462_data + (v428_data * (sycl::group_broadcast(item.get_sub_group(), v459_data, 2))));
              }
              if (v9_lead < 12) {
                float v468_data = r3[3];
                float v469_data = r2[0];
                float v472_data = ir4[0];
                ir4[0] = (v472_data + (v468_data * (sycl::group_broadcast(item.get_sub_group(), v469_data, 3))));
                float v475_data = r2[1];
                float v478_data = ir4[1];
                ir4[1] = (v478_data + (v468_data * (sycl::group_broadcast(item.get_sub_group(), v475_data, 3))));
                float v481_data = r2[2];
                float v484_data = ir4[2];
                ir4[2] = (v484_data + (v468_data * (sycl::group_broadcast(item.get_sub_group(), v481_data, 3))));
                float v487_data = r2[3];
                float v490_data = ir4[3];
                ir4[3] = (v490_data + (v468_data * (sycl::group_broadcast(item.get_sub_group(), v487_data, 3))));
                float v493_data = r2[4];
                float v496_data = ir4[4];
                ir4[4] = (v496_data + (v468_data * (sycl::group_broadcast(item.get_sub_group(), v493_data, 3))));
                float v499_data = r2[5];
                float v502_data = ir4[5];
                ir4[5] = (v502_data + (v468_data * (sycl::group_broadcast(item.get_sub_group(), v499_data, 3))));
              }
              if (v9_lead < 12) {
                float v508_data = r3[4];
                float v509_data = r2[0];
                float v512_data = ir4[0];
                ir4[0] = (v512_data + (v508_data * (sycl::group_broadcast(item.get_sub_group(), v509_data, 4))));
                float v515_data = r2[1];
                float v518_data = ir4[1];
                ir4[1] = (v518_data + (v508_data * (sycl::group_broadcast(item.get_sub_group(), v515_data, 4))));
                float v521_data = r2[2];
                float v524_data = ir4[2];
                ir4[2] = (v524_data + (v508_data * (sycl::group_broadcast(item.get_sub_group(), v521_data, 4))));
                float v527_data = r2[3];
                float v530_data = ir4[3];
                ir4[3] = (v530_data + (v508_data * (sycl::group_broadcast(item.get_sub_group(), v527_data, 4))));
                float v533_data = r2[4];
                float v536_data = ir4[4];
                ir4[4] = (v536_data + (v508_data * (sycl::group_broadcast(item.get_sub_group(), v533_data, 4))));
                float v539_data = r2[5];
                float v542_data = ir4[5];
                ir4[5] = (v542_data + (v508_data * (sycl::group_broadcast(item.get_sub_group(), v539_data, 4))));
              }
              if (v9_lead < 12) {
                float v548_data = r3[5];
                float v549_data = r2[0];
                float v552_data = ir4[0];
                ir4[0] = (v552_data + (v548_data * (sycl::group_broadcast(item.get_sub_group(), v549_data, 5))));
                float v555_data = r2[1];
                float v558_data = ir4[1];
                ir4[1] = (v558_data + (v548_data * (sycl::group_broadcast(item.get_sub_group(), v555_data, 5))));
                float v561_data = r2[2];
                float v564_data = ir4[2];
                ir4[2] = (v564_data + (v548_data * (sycl::group_broadcast(item.get_sub_group(), v561_data, 5))));
                float v567_data = r2[3];
                float v570_data = ir4[3];
                ir4[3] = (v570_data + (v548_data * (sycl::group_broadcast(item.get_sub_group(), v567_data, 5))));
                float v573_data = r2[4];
                float v576_data = ir4[4];
                ir4[4] = (v576_data + (v548_data * (sycl::group_broadcast(item.get_sub_group(), v573_data, 5))));
                float v579_data = r2[5];
                float v582_data = ir4[5];
                ir4[5] = (v582_data + (v548_data * (sycl::group_broadcast(item.get_sub_group(), v579_data, 5))));
              }
              if (v9_lead < 12) {
                float v588_data = r3[6];
                float v589_data = r2[0];
                float v592_data = ir4[0];
                ir4[0] = (v592_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v589_data, 6))));
                float v595_data = r2[1];
                float v598_data = ir4[1];
                ir4[1] = (v598_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v595_data, 6))));
                float v601_data = r2[2];
                float v604_data = ir4[2];
                ir4[2] = (v604_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v601_data, 6))));
                float v607_data = r2[3];
                float v610_data = ir4[3];
                ir4[3] = (v610_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v607_data, 6))));
                float v613_data = r2[4];
                float v616_data = ir4[4];
                ir4[4] = (v616_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v613_data, 6))));
                float v619_data = r2[5];
                float v622_data = ir4[5];
                ir4[5] = (v622_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v619_data, 6))));
              }
              if (v9_lead < 12) {
                float v628_data = r3[7];
                float v629_data = r2[0];
                float v632_data = ir4[0];
                ir4[0] = (v632_data + (v628_data * (sycl::group_broadcast(item.get_sub_group(), v629_data, 7))));
                float v635_data = r2[1];
                float v638_data = ir4[1];
                ir4[1] = (v638_data + (v628_data * (sycl::group_broadcast(item.get_sub_group(), v635_data, 7))));
                float v641_data = r2[2];
                float v644_data = ir4[2];
                ir4[2] = (v644_data + (v628_data * (sycl::group_broadcast(item.get_sub_group(), v641_data, 7))));
                float v647_data = r2[3];
                float v650_data = ir4[3];
                ir4[3] = (v650_data + (v628_data * (sycl::group_broadcast(item.get_sub_group(), v647_data, 7))));
                float v653_data = r2[4];
                float v656_data = ir4[4];
                ir4[4] = (v656_data + (v628_data * (sycl::group_broadcast(item.get_sub_group(), v653_data, 7))));
                float v659_data = r2[5];
                float v662_data = ir4[5];
                ir4[5] = (v662_data + (v628_data * (sycl::group_broadcast(item.get_sub_group(), v659_data, 7))));
              }
              if (v9_lead < 12) {
                float v668_data = r3[8];
                float v669_data = r2[0];
                float v672_data = ir4[0];
                ir4[0] = (v672_data + (v668_data * (sycl::group_broadcast(item.get_sub_group(), v669_data, 8))));
                float v675_data = r2[1];
                float v678_data = ir4[1];
                ir4[1] = (v678_data + (v668_data * (sycl::group_broadcast(item.get_sub_group(), v675_data, 8))));
                float v681_data = r2[2];
                float v684_data = ir4[2];
                ir4[2] = (v684_data + (v668_data * (sycl::group_broadcast(item.get_sub_group(), v681_data, 8))));
                float v687_data = r2[3];
                float v690_data = ir4[3];
                ir4[3] = (v690_data + (v668_data * (sycl::group_broadcast(item.get_sub_group(), v687_data, 8))));
                float v693_data = r2[4];
                float v696_data = ir4[4];
                ir4[4] = (v696_data + (v668_data * (sycl::group_broadcast(item.get_sub_group(), v693_data, 8))));
                float v699_data = r2[5];
                float v702_data = ir4[5];
                ir4[5] = (v702_data + (v668_data * (sycl::group_broadcast(item.get_sub_group(), v699_data, 8))));
              }
              if (v9_lead < 12) {
                float v708_data = r3[9];
                float v709_data = r2[0];
                float v712_data = ir4[0];
                ir4[0] = (v712_data + (v708_data * (sycl::group_broadcast(item.get_sub_group(), v709_data, 9))));
                float v715_data = r2[1];
                float v718_data = ir4[1];
                ir4[1] = (v718_data + (v708_data * (sycl::group_broadcast(item.get_sub_group(), v715_data, 9))));
                float v721_data = r2[2];
                float v724_data = ir4[2];
                ir4[2] = (v724_data + (v708_data * (sycl::group_broadcast(item.get_sub_group(), v721_data, 9))));
                float v727_data = r2[3];
                float v730_data = ir4[3];
                ir4[3] = (v730_data + (v708_data * (sycl::group_broadcast(item.get_sub_group(), v727_data, 9))));
                float v733_data = r2[4];
                float v736_data = ir4[4];
                ir4[4] = (v736_data + (v708_data * (sycl::group_broadcast(item.get_sub_group(), v733_data, 9))));
                float v739_data = r2[5];
                float v742_data = ir4[5];
                ir4[5] = (v742_data + (v708_data * (sycl::group_broadcast(item.get_sub_group(), v739_data, 9))));
              }
              if (v9_lead < 12) {
                float v748_data = r3[10];
                float v749_data = r2[0];
                float v752_data = ir4[0];
                ir4[0] = (v752_data + (v748_data * (sycl::group_broadcast(item.get_sub_group(), v749_data, 10))));
                float v755_data = r2[1];
                float v758_data = ir4[1];
                ir4[1] = (v758_data + (v748_data * (sycl::group_broadcast(item.get_sub_group(), v755_data, 10))));
                float v761_data = r2[2];
                float v764_data = ir4[2];
                ir4[2] = (v764_data + (v748_data * (sycl::group_broadcast(item.get_sub_group(), v761_data, 10))));
                float v767_data = r2[3];
                float v770_data = ir4[3];
                ir4[3] = (v770_data + (v748_data * (sycl::group_broadcast(item.get_sub_group(), v767_data, 10))));
                float v773_data = r2[4];
                float v776_data = ir4[4];
                ir4[4] = (v776_data + (v748_data * (sycl::group_broadcast(item.get_sub_group(), v773_data, 10))));
                float v779_data = r2[5];
                float v782_data = ir4[5];
                ir4[5] = (v782_data + (v748_data * (sycl::group_broadcast(item.get_sub_group(), v779_data, 10))));
              }
              if (v9_lead < 12) {
                float v788_data = r3[11];
                float v789_data = r2[0];
                float v792_data = ir4[0];
                ir4[0] = (v792_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v789_data, 11))));
                float v795_data = r2[1];
                float v798_data = ir4[1];
                ir4[1] = (v798_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v795_data, 11))));
                float v801_data = r2[2];
                float v804_data = ir4[2];
                ir4[2] = (v804_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v801_data, 11))));
                float v807_data = r2[3];
                float v810_data = ir4[3];
                ir4[3] = (v810_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v807_data, 11))));
                float v813_data = r2[4];
                float v816_data = ir4[4];
                ir4[4] = (v816_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v813_data, 11))));
                float v819_data = r2[5];
                float v822_data = ir4[5];
                ir4[5] = (v822_data + (v788_data * (sycl::group_broadcast(item.get_sub_group(), v819_data, 11))));
              }
              if (v9_lead < 12) {
                #pragma unroll
                for (int32_t v828_n1 = 0; v828_n1 < 6; ++v828_n1) {
                  float v830_data = ir4[v828_n1];
                  r4[v828_n1] = v830_data;
                }
              }
              // glb_m2 = store{r>g}(r4);
              if (v9_lead < 12) {
                #pragma unroll
                for (int32_t v836_i1 = 0; v836_i1 < 6; ++v836_i1) {
                  float v838_data = r4[v836_i1];
                  glb_m2[(v9_lead + (v836_i1 * 12))] = v838_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

