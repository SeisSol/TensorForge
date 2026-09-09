// === base name ===
kernel_6dbf36423ff9f407

// === header ===
void launcher_kernel_6dbf36423ff9f407(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_6dbf36423ff9f407(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_6dbf36423ff9f407(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_6dbf36423ff9f407(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[batchId0 * 46 + 0 + m2_extraOffset];
              double r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v12_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
                int32_t v19_lead = v12_lead + (v13_i0 * 16);
                #pragma unroll
                for (int32_t v14_i1 = 0; v14_i1 < 16; ++v14_i1) {
                  double v22_data = glb_m1[(v19_lead + (v14_i1 * 16))];
                  r0[(v13_i0 + v14_i1)] = v22_data;
                }
              }
              double r1[16]{};
              // r1 = load{g>r}(glb_m2);
              double v25_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v25_lin;
              double v26_lin = glb_m2[16 + item.get_local_id(0) * 1];
              r1[1] = v26_lin;
              double v27_lin = glb_m2[32 + item.get_local_id(0) * 1];
              r1[2] = v27_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              double r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              double ir2[16]{};
              double v33_data = r0[0];
              double v34_data = r1[0];
              double v36_data = ir2[0];
              ir2[0] = (v36_data + (v33_data * v34_data));
              double v39_data = r1[2];
              double v41_data = ir2[1];
              ir2[1] = (v41_data + (v33_data * v39_data));
              double v60_data = r0[1];
              double v61_data = r1[1];
              double v63_data = ir2[0];
              ir2[0] = (v63_data + (v60_data * v61_data));
              double v66_data = r1[3];
              double v68_data = ir2[1];
              ir2[1] = (v68_data + (v60_data * v66_data));
              double v71_data = r1[5];
              double v73_data = ir2[2];
              ir2[2] = (v73_data + (v60_data * v71_data));
              double v91_data = r0[2];
              double v93_data = r1[4];
              double v95_data = ir2[1];
              ir2[1] = (v95_data + (v91_data * v93_data));
              double v98_data = r1[6];
              double v100_data = ir2[2];
              ir2[2] = (v100_data + (v91_data * v98_data));
              double v103_data = r1[8];
              double v105_data = ir2[3];
              ir2[3] = (v105_data + (v91_data * v103_data));
              double v122_data = r0[3];
              double v125_data = r1[7];
              double v127_data = ir2[2];
              ir2[2] = (v127_data + (v122_data * v125_data));
              double v130_data = r1[9];
              double v132_data = ir2[3];
              ir2[3] = (v132_data + (v122_data * v130_data));
              double v135_data = r1[11];
              double v137_data = ir2[4];
              ir2[4] = (v137_data + (v122_data * v135_data));
              double v153_data = r0[4];
              double v157_data = r1[10];
              double v159_data = ir2[3];
              ir2[3] = (v159_data + (v153_data * v157_data));
              double v162_data = r1[12];
              double v164_data = ir2[4];
              ir2[4] = (v164_data + (v153_data * v162_data));
              double v167_data = r1[14];
              double v169_data = ir2[5];
              ir2[5] = (v169_data + (v153_data * v167_data));
              double v184_data = r0[5];
              double v189_data = r1[13];
              double v191_data = ir2[4];
              ir2[4] = (v191_data + (v184_data * v189_data));
              double v194_data = r1[15];
              double v196_data = ir2[5];
              ir2[5] = (v196_data + (v184_data * v194_data));
              double v199_data = r1[17];
              double v201_data = ir2[6];
              ir2[6] = (v201_data + (v184_data * v199_data));
              double v215_data = r0[6];
              double v221_data = r1[16];
              double v223_data = ir2[5];
              ir2[5] = (v223_data + (v215_data * v221_data));
              double v226_data = r1[18];
              double v228_data = ir2[6];
              ir2[6] = (v228_data + (v215_data * v226_data));
              double v231_data = r1[20];
              double v233_data = ir2[7];
              ir2[7] = (v233_data + (v215_data * v231_data));
              double v246_data = r0[7];
              double v253_data = r1[19];
              double v255_data = ir2[6];
              ir2[6] = (v255_data + (v246_data * v253_data));
              double v258_data = r1[21];
              double v260_data = ir2[7];
              ir2[7] = (v260_data + (v246_data * v258_data));
              double v263_data = r1[23];
              double v265_data = ir2[8];
              ir2[8] = (v265_data + (v246_data * v263_data));
              double v277_data = r0[8];
              double v285_data = r1[22];
              double v287_data = ir2[7];
              ir2[7] = (v287_data + (v277_data * v285_data));
              double v290_data = r1[24];
              double v292_data = ir2[8];
              ir2[8] = (v292_data + (v277_data * v290_data));
              double v295_data = r1[26];
              double v297_data = ir2[9];
              ir2[9] = (v297_data + (v277_data * v295_data));
              double v308_data = r0[9];
              double v317_data = r1[25];
              double v319_data = ir2[8];
              ir2[8] = (v319_data + (v308_data * v317_data));
              double v322_data = r1[27];
              double v324_data = ir2[9];
              ir2[9] = (v324_data + (v308_data * v322_data));
              double v327_data = r1[29];
              double v329_data = ir2[10];
              ir2[10] = (v329_data + (v308_data * v327_data));
              double v339_data = r0[10];
              double v349_data = r1[28];
              double v351_data = ir2[9];
              ir2[9] = (v351_data + (v339_data * v349_data));
              double v354_data = r1[30];
              double v356_data = ir2[10];
              ir2[10] = (v356_data + (v339_data * v354_data));
              double v359_data = r1[32];
              double v361_data = ir2[11];
              ir2[11] = (v361_data + (v339_data * v359_data));
              double v370_data = r0[11];
              double v381_data = r1[31];
              double v383_data = ir2[10];
              ir2[10] = (v383_data + (v370_data * v381_data));
              double v386_data = r1[33];
              double v388_data = ir2[11];
              ir2[11] = (v388_data + (v370_data * v386_data));
              double v391_data = r1[35];
              double v393_data = ir2[12];
              ir2[12] = (v393_data + (v370_data * v391_data));
              double v401_data = r0[12];
              double v413_data = r1[34];
              double v415_data = ir2[11];
              ir2[11] = (v415_data + (v401_data * v413_data));
              double v418_data = r1[36];
              double v420_data = ir2[12];
              ir2[12] = (v420_data + (v401_data * v418_data));
              double v423_data = r1[38];
              double v425_data = ir2[13];
              ir2[13] = (v425_data + (v401_data * v423_data));
              double v432_data = r0[13];
              double v445_data = r1[37];
              double v447_data = ir2[12];
              ir2[12] = (v447_data + (v432_data * v445_data));
              double v450_data = r1[39];
              double v452_data = ir2[13];
              ir2[13] = (v452_data + (v432_data * v450_data));
              double v455_data = r1[41];
              double v457_data = ir2[14];
              ir2[14] = (v457_data + (v432_data * v455_data));
              double v463_data = r0[14];
              double v477_data = r1[40];
              double v479_data = ir2[13];
              ir2[13] = (v479_data + (v463_data * v477_data));
              double v482_data = r1[42];
              double v484_data = ir2[14];
              ir2[14] = (v484_data + (v463_data * v482_data));
              double v487_data = r1[44];
              double v489_data = ir2[15];
              ir2[15] = (v489_data + (v463_data * v487_data));
              double v494_data = r0[15];
              double v509_data = r1[43];
              double v511_data = ir2[14];
              ir2[14] = (v511_data + (v494_data * v509_data));
              double v514_data = r1[45];
              double v516_data = ir2[15];
              ir2[15] = (v516_data + (v494_data * v514_data));
              #pragma unroll
              for (int32_t v521_n0 = 0; v521_n0 < 1; ++v521_n0) {
                #pragma unroll
                for (int32_t v522_n1 = 0; v522_n1 < 16; ++v522_n1) {
                  int32_t v523_a = v521_n0 + v522_n1;
                  double v524_data = ir2[v523_a];
                  r2[v523_a] = v524_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v529_i0 = 0; v529_i0 < 1; ++v529_i0) {
                int32_t v537_lead = v12_lead + (v529_i0 * 16);
                #pragma unroll
                for (int32_t v530_i1 = 0; v530_i1 < 16; ++v530_i1) {
                  double v532_data = r2[(v529_i0 + v530_i1)];
                  glb_m0[(v537_lead + (v530_i1 * 16))] = v532_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

