// === base name ===
kernel_30948bd44e

// === header ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_30948bd44e(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_30948bd44e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×16(16×16) {0..16}×{0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m2 16×16(16×16) {0..16}×{0..16} strided
        // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
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
              float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
                int32_t v15_lead = v8_lead + (v9_i0 * 16);
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 16; ++v10_i1) {
                  float v18_data = glb_m1[(v15_lead + (v10_i1 * 16))];
                  r0[(v9_i0 + v10_i1)] = v18_data;
                }
              }
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              float v21_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v21_lin;
              float v22_lin = glb_m2[16 + item.get_local_id(0) * 1];
              r1[1] = v22_lin;
              float v23_lin = glb_m2[32 + item.get_local_id(0) * 1];
              r1[2] = v23_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir2[16]{};
              float v29_data = r0[0];
              float v30_data = r1[0];
              float v32_data = ir2[0];
              ir2[0] = (v32_data + (v29_data * v30_data));
              float v35_data = r1[2];
              float v37_data = ir2[1];
              ir2[1] = (v37_data + (v29_data * v35_data));
              float v56_data = r0[1];
              float v57_data = r1[1];
              float v59_data = ir2[0];
              ir2[0] = (v59_data + (v56_data * v57_data));
              float v62_data = r1[3];
              float v64_data = ir2[1];
              ir2[1] = (v64_data + (v56_data * v62_data));
              float v67_data = r1[5];
              float v69_data = ir2[2];
              ir2[2] = (v69_data + (v56_data * v67_data));
              float v87_data = r0[2];
              float v89_data = r1[4];
              float v91_data = ir2[1];
              ir2[1] = (v91_data + (v87_data * v89_data));
              float v94_data = r1[6];
              float v96_data = ir2[2];
              ir2[2] = (v96_data + (v87_data * v94_data));
              float v99_data = r1[8];
              float v101_data = ir2[3];
              ir2[3] = (v101_data + (v87_data * v99_data));
              float v118_data = r0[3];
              float v121_data = r1[7];
              float v123_data = ir2[2];
              ir2[2] = (v123_data + (v118_data * v121_data));
              float v126_data = r1[9];
              float v128_data = ir2[3];
              ir2[3] = (v128_data + (v118_data * v126_data));
              float v131_data = r1[11];
              float v133_data = ir2[4];
              ir2[4] = (v133_data + (v118_data * v131_data));
              float v149_data = r0[4];
              float v153_data = r1[10];
              float v155_data = ir2[3];
              ir2[3] = (v155_data + (v149_data * v153_data));
              float v158_data = r1[12];
              float v160_data = ir2[4];
              ir2[4] = (v160_data + (v149_data * v158_data));
              float v163_data = r1[14];
              float v165_data = ir2[5];
              ir2[5] = (v165_data + (v149_data * v163_data));
              float v180_data = r0[5];
              float v185_data = r1[13];
              float v187_data = ir2[4];
              ir2[4] = (v187_data + (v180_data * v185_data));
              float v190_data = r1[15];
              float v192_data = ir2[5];
              ir2[5] = (v192_data + (v180_data * v190_data));
              float v195_data = r1[17];
              float v197_data = ir2[6];
              ir2[6] = (v197_data + (v180_data * v195_data));
              float v211_data = r0[6];
              float v217_data = r1[16];
              float v219_data = ir2[5];
              ir2[5] = (v219_data + (v211_data * v217_data));
              float v222_data = r1[18];
              float v224_data = ir2[6];
              ir2[6] = (v224_data + (v211_data * v222_data));
              float v227_data = r1[20];
              float v229_data = ir2[7];
              ir2[7] = (v229_data + (v211_data * v227_data));
              float v242_data = r0[7];
              float v249_data = r1[19];
              float v251_data = ir2[6];
              ir2[6] = (v251_data + (v242_data * v249_data));
              float v254_data = r1[21];
              float v256_data = ir2[7];
              ir2[7] = (v256_data + (v242_data * v254_data));
              float v259_data = r1[23];
              float v261_data = ir2[8];
              ir2[8] = (v261_data + (v242_data * v259_data));
              float v273_data = r0[8];
              float v281_data = r1[22];
              float v283_data = ir2[7];
              ir2[7] = (v283_data + (v273_data * v281_data));
              float v286_data = r1[24];
              float v288_data = ir2[8];
              ir2[8] = (v288_data + (v273_data * v286_data));
              float v291_data = r1[26];
              float v293_data = ir2[9];
              ir2[9] = (v293_data + (v273_data * v291_data));
              float v304_data = r0[9];
              float v313_data = r1[25];
              float v315_data = ir2[8];
              ir2[8] = (v315_data + (v304_data * v313_data));
              float v318_data = r1[27];
              float v320_data = ir2[9];
              ir2[9] = (v320_data + (v304_data * v318_data));
              float v323_data = r1[29];
              float v325_data = ir2[10];
              ir2[10] = (v325_data + (v304_data * v323_data));
              float v335_data = r0[10];
              float v345_data = r1[28];
              float v347_data = ir2[9];
              ir2[9] = (v347_data + (v335_data * v345_data));
              float v350_data = r1[30];
              float v352_data = ir2[10];
              ir2[10] = (v352_data + (v335_data * v350_data));
              float v355_data = r1[32];
              float v357_data = ir2[11];
              ir2[11] = (v357_data + (v335_data * v355_data));
              float v366_data = r0[11];
              float v377_data = r1[31];
              float v379_data = ir2[10];
              ir2[10] = (v379_data + (v366_data * v377_data));
              float v382_data = r1[33];
              float v384_data = ir2[11];
              ir2[11] = (v384_data + (v366_data * v382_data));
              float v387_data = r1[35];
              float v389_data = ir2[12];
              ir2[12] = (v389_data + (v366_data * v387_data));
              float v397_data = r0[12];
              float v409_data = r1[34];
              float v411_data = ir2[11];
              ir2[11] = (v411_data + (v397_data * v409_data));
              float v414_data = r1[36];
              float v416_data = ir2[12];
              ir2[12] = (v416_data + (v397_data * v414_data));
              float v419_data = r1[38];
              float v421_data = ir2[13];
              ir2[13] = (v421_data + (v397_data * v419_data));
              float v428_data = r0[13];
              float v441_data = r1[37];
              float v443_data = ir2[12];
              ir2[12] = (v443_data + (v428_data * v441_data));
              float v446_data = r1[39];
              float v448_data = ir2[13];
              ir2[13] = (v448_data + (v428_data * v446_data));
              float v451_data = r1[41];
              float v453_data = ir2[14];
              ir2[14] = (v453_data + (v428_data * v451_data));
              float v459_data = r0[14];
              float v473_data = r1[40];
              float v475_data = ir2[13];
              ir2[13] = (v475_data + (v459_data * v473_data));
              float v478_data = r1[42];
              float v480_data = ir2[14];
              ir2[14] = (v480_data + (v459_data * v478_data));
              float v483_data = r1[44];
              float v485_data = ir2[15];
              ir2[15] = (v485_data + (v459_data * v483_data));
              float v490_data = r0[15];
              float v505_data = r1[43];
              float v507_data = ir2[14];
              ir2[14] = (v507_data + (v490_data * v505_data));
              float v510_data = r1[45];
              float v512_data = ir2[15];
              ir2[15] = (v512_data + (v490_data * v510_data));
              #pragma unroll
              for (int32_t v517_n0 = 0; v517_n0 < 1; ++v517_n0) {
                #pragma unroll
                for (int32_t v518_n1 = 0; v518_n1 < 16; ++v518_n1) {
                  int32_t v519_a = v517_n0 + v518_n1;
                  float v520_data = ir2[v519_a];
                  r2[v519_a] = v520_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v525_i0 = 0; v525_i0 < 1; ++v525_i0) {
                int32_t v533_lead = v8_lead + (v525_i0 * 16);
                #pragma unroll
                for (int32_t v526_i1 = 0; v526_i1 < 16; ++v526_i1) {
                  float v528_data = r2[(v525_i0 + v526_i1)];
                  glb_m0[(v533_lead + (v526_i1 * 16))] = v528_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

