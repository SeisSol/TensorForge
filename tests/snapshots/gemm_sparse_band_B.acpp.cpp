// === base name ===
kernel_bfddc0b2645d2ba3

// === header ===
void launcher_kernel_bfddc0b2645d2ba3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_bfddc0b2645d2ba3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_bfddc0b2645d2ba3(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_bfddc0b2645d2ba3(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
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
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 46 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v12_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
                int32_t v19_lead = v12_lead + (v13_i0 * 16);
                #pragma unroll
                for (int32_t v14_i1 = 0; v14_i1 < 16; ++v14_i1) {
                  float v22_data = glb_m1[(v19_lead + (v14_i1 * 16))];
                  r0[(v13_i0 + v14_i1)] = v22_data;
                }
              }
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              float v25_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v25_lin;
              float v26_lin = glb_m2[16 + item.get_local_id(0) * 1];
              r1[1] = v26_lin;
              float v27_lin = glb_m2[32 + item.get_local_id(0) * 1];
              r1[2] = v27_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir2[16]{};
              float v33_data = r0[0];
              float v34_data = r1[0];
              float v36_data = ir2[0];
              ir2[0] = (v36_data + (v33_data * v34_data));
              float v39_data = r1[2];
              float v41_data = ir2[1];
              ir2[1] = (v41_data + (v33_data * v39_data));
              float v60_data = r0[1];
              float v61_data = r1[1];
              float v63_data = ir2[0];
              ir2[0] = (v63_data + (v60_data * v61_data));
              float v66_data = r1[3];
              float v68_data = ir2[1];
              ir2[1] = (v68_data + (v60_data * v66_data));
              float v71_data = r1[5];
              float v73_data = ir2[2];
              ir2[2] = (v73_data + (v60_data * v71_data));
              float v91_data = r0[2];
              float v93_data = r1[4];
              float v95_data = ir2[1];
              ir2[1] = (v95_data + (v91_data * v93_data));
              float v98_data = r1[6];
              float v100_data = ir2[2];
              ir2[2] = (v100_data + (v91_data * v98_data));
              float v103_data = r1[8];
              float v105_data = ir2[3];
              ir2[3] = (v105_data + (v91_data * v103_data));
              float v122_data = r0[3];
              float v125_data = r1[7];
              float v127_data = ir2[2];
              ir2[2] = (v127_data + (v122_data * v125_data));
              float v130_data = r1[9];
              float v132_data = ir2[3];
              ir2[3] = (v132_data + (v122_data * v130_data));
              float v135_data = r1[11];
              float v137_data = ir2[4];
              ir2[4] = (v137_data + (v122_data * v135_data));
              float v153_data = r0[4];
              float v157_data = r1[10];
              float v159_data = ir2[3];
              ir2[3] = (v159_data + (v153_data * v157_data));
              float v162_data = r1[12];
              float v164_data = ir2[4];
              ir2[4] = (v164_data + (v153_data * v162_data));
              float v167_data = r1[14];
              float v169_data = ir2[5];
              ir2[5] = (v169_data + (v153_data * v167_data));
              float v184_data = r0[5];
              float v189_data = r1[13];
              float v191_data = ir2[4];
              ir2[4] = (v191_data + (v184_data * v189_data));
              float v194_data = r1[15];
              float v196_data = ir2[5];
              ir2[5] = (v196_data + (v184_data * v194_data));
              float v199_data = r1[17];
              float v201_data = ir2[6];
              ir2[6] = (v201_data + (v184_data * v199_data));
              float v215_data = r0[6];
              float v221_data = r1[16];
              float v223_data = ir2[5];
              ir2[5] = (v223_data + (v215_data * v221_data));
              float v226_data = r1[18];
              float v228_data = ir2[6];
              ir2[6] = (v228_data + (v215_data * v226_data));
              float v231_data = r1[20];
              float v233_data = ir2[7];
              ir2[7] = (v233_data + (v215_data * v231_data));
              float v246_data = r0[7];
              float v253_data = r1[19];
              float v255_data = ir2[6];
              ir2[6] = (v255_data + (v246_data * v253_data));
              float v258_data = r1[21];
              float v260_data = ir2[7];
              ir2[7] = (v260_data + (v246_data * v258_data));
              float v263_data = r1[23];
              float v265_data = ir2[8];
              ir2[8] = (v265_data + (v246_data * v263_data));
              float v277_data = r0[8];
              float v285_data = r1[22];
              float v287_data = ir2[7];
              ir2[7] = (v287_data + (v277_data * v285_data));
              float v290_data = r1[24];
              float v292_data = ir2[8];
              ir2[8] = (v292_data + (v277_data * v290_data));
              float v295_data = r1[26];
              float v297_data = ir2[9];
              ir2[9] = (v297_data + (v277_data * v295_data));
              float v308_data = r0[9];
              float v317_data = r1[25];
              float v319_data = ir2[8];
              ir2[8] = (v319_data + (v308_data * v317_data));
              float v322_data = r1[27];
              float v324_data = ir2[9];
              ir2[9] = (v324_data + (v308_data * v322_data));
              float v327_data = r1[29];
              float v329_data = ir2[10];
              ir2[10] = (v329_data + (v308_data * v327_data));
              float v339_data = r0[10];
              float v349_data = r1[28];
              float v351_data = ir2[9];
              ir2[9] = (v351_data + (v339_data * v349_data));
              float v354_data = r1[30];
              float v356_data = ir2[10];
              ir2[10] = (v356_data + (v339_data * v354_data));
              float v359_data = r1[32];
              float v361_data = ir2[11];
              ir2[11] = (v361_data + (v339_data * v359_data));
              float v370_data = r0[11];
              float v381_data = r1[31];
              float v383_data = ir2[10];
              ir2[10] = (v383_data + (v370_data * v381_data));
              float v386_data = r1[33];
              float v388_data = ir2[11];
              ir2[11] = (v388_data + (v370_data * v386_data));
              float v391_data = r1[35];
              float v393_data = ir2[12];
              ir2[12] = (v393_data + (v370_data * v391_data));
              float v401_data = r0[12];
              float v413_data = r1[34];
              float v415_data = ir2[11];
              ir2[11] = (v415_data + (v401_data * v413_data));
              float v418_data = r1[36];
              float v420_data = ir2[12];
              ir2[12] = (v420_data + (v401_data * v418_data));
              float v423_data = r1[38];
              float v425_data = ir2[13];
              ir2[13] = (v425_data + (v401_data * v423_data));
              float v432_data = r0[13];
              float v445_data = r1[37];
              float v447_data = ir2[12];
              ir2[12] = (v447_data + (v432_data * v445_data));
              float v450_data = r1[39];
              float v452_data = ir2[13];
              ir2[13] = (v452_data + (v432_data * v450_data));
              float v455_data = r1[41];
              float v457_data = ir2[14];
              ir2[14] = (v457_data + (v432_data * v455_data));
              float v463_data = r0[14];
              float v477_data = r1[40];
              float v479_data = ir2[13];
              ir2[13] = (v479_data + (v463_data * v477_data));
              float v482_data = r1[42];
              float v484_data = ir2[14];
              ir2[14] = (v484_data + (v463_data * v482_data));
              float v487_data = r1[44];
              float v489_data = ir2[15];
              ir2[15] = (v489_data + (v463_data * v487_data));
              float v494_data = r0[15];
              float v509_data = r1[43];
              float v511_data = ir2[14];
              ir2[14] = (v511_data + (v494_data * v509_data));
              float v514_data = r1[45];
              float v516_data = ir2[15];
              ir2[15] = (v516_data + (v494_data * v514_data));
              #pragma unroll
              for (int32_t v521_n0 = 0; v521_n0 < 1; ++v521_n0) {
                #pragma unroll
                for (int32_t v522_n1 = 0; v522_n1 < 16; ++v522_n1) {
                  int32_t v523_a = v521_n0 + v522_n1;
                  float v524_data = ir2[v523_a];
                  r2[v523_a] = v524_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v529_i0 = 0; v529_i0 < 1; ++v529_i0) {
                int32_t v537_lead = v12_lead + (v529_i0 * 16);
                #pragma unroll
                for (int32_t v530_i1 = 0; v530_i1 < 16; ++v530_i1) {
                  float v532_data = r2[(v529_i0 + v530_i1)];
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

