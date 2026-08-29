// === base name ===
kernel_417e1ddcc4

// === header ===
void launcher_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_417e1ddcc4(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_417e1ddcc4(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
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
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          double* tempShrMem = &localShrMem0[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
              double r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
                int32_t v15_lead = v8_lead + (v9_i0 * 16);
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 16; ++v10_i1) {
                  double v18_data = glb_m1[(v15_lead + (v10_i1 * 16))];
                  r0[(v9_i0 + v10_i1)] = v18_data;
                }
              }
              double r1[16]{};
              // r1 = load{g>r}(glb_m2);
              double v21_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v21_lin;
              double v22_lin = glb_m2[16 + item.get_local_id(0) * 1];
              r1[1] = v22_lin;
              double v23_lin = glb_m2[32 + item.get_local_id(0) * 1];
              r1[2] = v23_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              double r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              double ir2[16]{};
              double v29_data = r0[0];
              double v30_data = r1[0];
              double v32_data = ir2[0];
              ir2[0] = (v32_data + (v29_data * v30_data));
              double v35_data = r1[2];
              double v37_data = ir2[1];
              ir2[1] = (v37_data + (v29_data * v35_data));
              double v56_data = r0[1];
              double v57_data = r1[1];
              double v59_data = ir2[0];
              ir2[0] = (v59_data + (v56_data * v57_data));
              double v62_data = r1[3];
              double v64_data = ir2[1];
              ir2[1] = (v64_data + (v56_data * v62_data));
              double v67_data = r1[5];
              double v69_data = ir2[2];
              ir2[2] = (v69_data + (v56_data * v67_data));
              double v87_data = r0[2];
              double v89_data = r1[4];
              double v91_data = ir2[1];
              ir2[1] = (v91_data + (v87_data * v89_data));
              double v94_data = r1[6];
              double v96_data = ir2[2];
              ir2[2] = (v96_data + (v87_data * v94_data));
              double v99_data = r1[8];
              double v101_data = ir2[3];
              ir2[3] = (v101_data + (v87_data * v99_data));
              double v118_data = r0[3];
              double v121_data = r1[7];
              double v123_data = ir2[2];
              ir2[2] = (v123_data + (v118_data * v121_data));
              double v126_data = r1[9];
              double v128_data = ir2[3];
              ir2[3] = (v128_data + (v118_data * v126_data));
              double v131_data = r1[11];
              double v133_data = ir2[4];
              ir2[4] = (v133_data + (v118_data * v131_data));
              double v149_data = r0[4];
              double v153_data = r1[10];
              double v155_data = ir2[3];
              ir2[3] = (v155_data + (v149_data * v153_data));
              double v158_data = r1[12];
              double v160_data = ir2[4];
              ir2[4] = (v160_data + (v149_data * v158_data));
              double v163_data = r1[14];
              double v165_data = ir2[5];
              ir2[5] = (v165_data + (v149_data * v163_data));
              double v180_data = r0[5];
              double v185_data = r1[13];
              double v187_data = ir2[4];
              ir2[4] = (v187_data + (v180_data * v185_data));
              double v190_data = r1[15];
              double v192_data = ir2[5];
              ir2[5] = (v192_data + (v180_data * v190_data));
              double v195_data = r1[17];
              double v197_data = ir2[6];
              ir2[6] = (v197_data + (v180_data * v195_data));
              double v211_data = r0[6];
              double v217_data = r1[16];
              double v219_data = ir2[5];
              ir2[5] = (v219_data + (v211_data * v217_data));
              double v222_data = r1[18];
              double v224_data = ir2[6];
              ir2[6] = (v224_data + (v211_data * v222_data));
              double v227_data = r1[20];
              double v229_data = ir2[7];
              ir2[7] = (v229_data + (v211_data * v227_data));
              double v242_data = r0[7];
              double v249_data = r1[19];
              double v251_data = ir2[6];
              ir2[6] = (v251_data + (v242_data * v249_data));
              double v254_data = r1[21];
              double v256_data = ir2[7];
              ir2[7] = (v256_data + (v242_data * v254_data));
              double v259_data = r1[23];
              double v261_data = ir2[8];
              ir2[8] = (v261_data + (v242_data * v259_data));
              double v273_data = r0[8];
              double v281_data = r1[22];
              double v283_data = ir2[7];
              ir2[7] = (v283_data + (v273_data * v281_data));
              double v286_data = r1[24];
              double v288_data = ir2[8];
              ir2[8] = (v288_data + (v273_data * v286_data));
              double v291_data = r1[26];
              double v293_data = ir2[9];
              ir2[9] = (v293_data + (v273_data * v291_data));
              double v304_data = r0[9];
              double v313_data = r1[25];
              double v315_data = ir2[8];
              ir2[8] = (v315_data + (v304_data * v313_data));
              double v318_data = r1[27];
              double v320_data = ir2[9];
              ir2[9] = (v320_data + (v304_data * v318_data));
              double v323_data = r1[29];
              double v325_data = ir2[10];
              ir2[10] = (v325_data + (v304_data * v323_data));
              double v335_data = r0[10];
              double v345_data = r1[28];
              double v347_data = ir2[9];
              ir2[9] = (v347_data + (v335_data * v345_data));
              double v350_data = r1[30];
              double v352_data = ir2[10];
              ir2[10] = (v352_data + (v335_data * v350_data));
              double v355_data = r1[32];
              double v357_data = ir2[11];
              ir2[11] = (v357_data + (v335_data * v355_data));
              double v366_data = r0[11];
              double v377_data = r1[31];
              double v379_data = ir2[10];
              ir2[10] = (v379_data + (v366_data * v377_data));
              double v382_data = r1[33];
              double v384_data = ir2[11];
              ir2[11] = (v384_data + (v366_data * v382_data));
              double v387_data = r1[35];
              double v389_data = ir2[12];
              ir2[12] = (v389_data + (v366_data * v387_data));
              double v397_data = r0[12];
              double v409_data = r1[34];
              double v411_data = ir2[11];
              ir2[11] = (v411_data + (v397_data * v409_data));
              double v414_data = r1[36];
              double v416_data = ir2[12];
              ir2[12] = (v416_data + (v397_data * v414_data));
              double v419_data = r1[38];
              double v421_data = ir2[13];
              ir2[13] = (v421_data + (v397_data * v419_data));
              double v428_data = r0[13];
              double v441_data = r1[37];
              double v443_data = ir2[12];
              ir2[12] = (v443_data + (v428_data * v441_data));
              double v446_data = r1[39];
              double v448_data = ir2[13];
              ir2[13] = (v448_data + (v428_data * v446_data));
              double v451_data = r1[41];
              double v453_data = ir2[14];
              ir2[14] = (v453_data + (v428_data * v451_data));
              double v459_data = r0[14];
              double v473_data = r1[40];
              double v475_data = ir2[13];
              ir2[13] = (v475_data + (v459_data * v473_data));
              double v478_data = r1[42];
              double v480_data = ir2[14];
              ir2[14] = (v480_data + (v459_data * v478_data));
              double v483_data = r1[44];
              double v485_data = ir2[15];
              ir2[15] = (v485_data + (v459_data * v483_data));
              double v490_data = r0[15];
              double v505_data = r1[43];
              double v507_data = ir2[14];
              ir2[14] = (v507_data + (v490_data * v505_data));
              double v510_data = r1[45];
              double v512_data = ir2[15];
              ir2[15] = (v512_data + (v490_data * v510_data));
              #pragma unroll
              for (int32_t v517_n0 = 0; v517_n0 < 1; ++v517_n0) {
                #pragma unroll
                for (int32_t v518_n1 = 0; v518_n1 < 16; ++v518_n1) {
                  int32_t v519_a = v517_n0 + v518_n1;
                  double v520_data = ir2[v519_a];
                  r2[v519_a] = v520_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v525_i0 = 0; v525_i0 < 1; ++v525_i0) {
                int32_t v533_lead = v8_lead + (v525_i0 * 16);
                #pragma unroll
                for (int32_t v526_i1 = 0; v526_i1 < 16; ++v526_i1) {
                  double v528_data = r2[(v525_i0 + v526_i1)];
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

