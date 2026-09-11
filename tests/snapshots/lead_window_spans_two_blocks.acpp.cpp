// === base name ===
kernel_64c7eadb91308ea7

// === header ===
void launcher_kernel_64c7eadb91308ea7(const float** m0, size_t m0_extraOffset, const float* m1, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_64c7eadb91308ea7(const float** m0, size_t m0_extraOffset, const float* m1, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_64c7eadb91308ea7(stream, grid, block,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_64c7eadb91308ea7(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, size_t m0_extraOffset, const float* m1, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 64×13(64×13) {0..64}×{0..13} pointer_based
        // m1 6(6) {0..6} none
        // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based
        // t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..64}×{0..13}×{0..6})[0, 1, 2] = m0 64×13(64×13) {0..64}×{0..13} pointer_based({0..64}×{0..13})[0, 1]×m1 6(6) {0..6} none({0..6})[2]
        // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based({0..15}×{0..1}×{0..6})[0, 1, 2] += t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..15}×{0..1}×{0..6})[0, 1, 2]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const float *const __restrict__ glb_m1 = &m1[0];
          for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v2_ahead1 = v1_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v1_batchId0][0 + m0_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v1_batchId0][0 + m2_extraOffset];
              float r0[26]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v14_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 2; ++v15_i0) {
                int32_t v21_lead = v14_lead + (v15_i0 * 32);
                #pragma unroll
                for (int32_t v16_i1 = 0; v16_i1 < 13; ++v16_i1) {
                  float v24_data = glb_m0[(v21_lead + (v16_i1 * 64))];
                  r0[(v15_i0 + (v16_i1 * 2))] = v24_data;
                }
              }
              float r2[12]{};
              // r2 = load{g>r}(glb_m2);
              if (v14_lead >= 20) {
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 1; ++v32_i1) {
                  int32_t v42_a = v14_lead + ((v32_i1 + 12) * 64);
                  int32_t v45_a = v32_i1 * 2;
                  #pragma unroll
                  for (int32_t v33_i2 = 0; v33_i2 < 6; ++v33_i2) {
                    float v44_data = glb_m2[(v42_a + (v33_i2 * 832))];
                    r2[(v45_a + (v33_i2 * 2))] = v44_data;
                  }
                }
              }
              if (v14_lead < 3) {
                int32_t v56_lead = v14_lead + 32_i32;
                #pragma unroll
                for (int32_t v50_i1 = 0; v50_i1 < 1; ++v50_i1) {
                  int32_t v60_a = v56_lead + ((v50_i1 + 12) * 64);
                  int32_t v65_a = 1 + (v50_i1 * 2);
                  #pragma unroll
                  for (int32_t v51_i2 = 0; v51_i2 < 6; ++v51_i2) {
                    float v62_data = glb_m2[(v60_a + (v51_i2 * 832))];
                    r2[(v65_a + (v51_i2 * 2))] = v62_data;
                  }
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[156]{};
              // r1 = +(r0 * glb_m1) + None
              // [(0, 64), (0, 13), (0, 6)] []
              float v71_data = r0[0];
              float v72_data = glb_m1[0];
              float v74_data = r1[0];
              r1[0] = (v74_data + (v71_data * v72_data));
              float v77_data = glb_m1[1];
              float v79_data = r1[26];
              r1[26] = (v79_data + (v71_data * v77_data));
              float v82_data = glb_m1[2];
              float v84_data = r1[52];
              r1[52] = (v84_data + (v71_data * v82_data));
              float v87_data = glb_m1[3];
              float v89_data = r1[78];
              r1[78] = (v89_data + (v71_data * v87_data));
              float v92_data = glb_m1[4];
              float v94_data = r1[104];
              r1[104] = (v94_data + (v71_data * v92_data));
              float v97_data = glb_m1[5];
              float v99_data = r1[130];
              r1[130] = (v99_data + (v71_data * v97_data));
              float v101_data = r0[2];
              float v104_data = r1[2];
              r1[2] = (v104_data + (v101_data * v72_data));
              float v109_data = r1[28];
              r1[28] = (v109_data + (v101_data * v77_data));
              float v114_data = r1[54];
              r1[54] = (v114_data + (v101_data * v82_data));
              float v119_data = r1[80];
              r1[80] = (v119_data + (v101_data * v87_data));
              float v124_data = r1[106];
              r1[106] = (v124_data + (v101_data * v92_data));
              float v129_data = r1[132];
              r1[132] = (v129_data + (v101_data * v97_data));
              float v131_data = r0[4];
              float v134_data = r1[4];
              r1[4] = (v134_data + (v131_data * v72_data));
              float v139_data = r1[30];
              r1[30] = (v139_data + (v131_data * v77_data));
              float v144_data = r1[56];
              r1[56] = (v144_data + (v131_data * v82_data));
              float v149_data = r1[82];
              r1[82] = (v149_data + (v131_data * v87_data));
              float v154_data = r1[108];
              r1[108] = (v154_data + (v131_data * v92_data));
              float v159_data = r1[134];
              r1[134] = (v159_data + (v131_data * v97_data));
              float v161_data = r0[6];
              float v164_data = r1[6];
              r1[6] = (v164_data + (v161_data * v72_data));
              float v169_data = r1[32];
              r1[32] = (v169_data + (v161_data * v77_data));
              float v174_data = r1[58];
              r1[58] = (v174_data + (v161_data * v82_data));
              float v179_data = r1[84];
              r1[84] = (v179_data + (v161_data * v87_data));
              float v184_data = r1[110];
              r1[110] = (v184_data + (v161_data * v92_data));
              float v189_data = r1[136];
              r1[136] = (v189_data + (v161_data * v97_data));
              float v191_data = r0[8];
              float v194_data = r1[8];
              r1[8] = (v194_data + (v191_data * v72_data));
              float v199_data = r1[34];
              r1[34] = (v199_data + (v191_data * v77_data));
              float v204_data = r1[60];
              r1[60] = (v204_data + (v191_data * v82_data));
              float v209_data = r1[86];
              r1[86] = (v209_data + (v191_data * v87_data));
              float v214_data = r1[112];
              r1[112] = (v214_data + (v191_data * v92_data));
              float v219_data = r1[138];
              r1[138] = (v219_data + (v191_data * v97_data));
              float v221_data = r0[10];
              float v224_data = r1[10];
              r1[10] = (v224_data + (v221_data * v72_data));
              float v229_data = r1[36];
              r1[36] = (v229_data + (v221_data * v77_data));
              float v234_data = r1[62];
              r1[62] = (v234_data + (v221_data * v82_data));
              float v239_data = r1[88];
              r1[88] = (v239_data + (v221_data * v87_data));
              float v244_data = r1[114];
              r1[114] = (v244_data + (v221_data * v92_data));
              float v249_data = r1[140];
              r1[140] = (v249_data + (v221_data * v97_data));
              float v251_data = r0[12];
              float v254_data = r1[12];
              r1[12] = (v254_data + (v251_data * v72_data));
              float v259_data = r1[38];
              r1[38] = (v259_data + (v251_data * v77_data));
              float v264_data = r1[64];
              r1[64] = (v264_data + (v251_data * v82_data));
              float v269_data = r1[90];
              r1[90] = (v269_data + (v251_data * v87_data));
              float v274_data = r1[116];
              r1[116] = (v274_data + (v251_data * v92_data));
              float v279_data = r1[142];
              r1[142] = (v279_data + (v251_data * v97_data));
              float v281_data = r0[14];
              float v284_data = r1[14];
              r1[14] = (v284_data + (v281_data * v72_data));
              float v289_data = r1[40];
              r1[40] = (v289_data + (v281_data * v77_data));
              float v294_data = r1[66];
              r1[66] = (v294_data + (v281_data * v82_data));
              float v299_data = r1[92];
              r1[92] = (v299_data + (v281_data * v87_data));
              float v304_data = r1[118];
              r1[118] = (v304_data + (v281_data * v92_data));
              float v309_data = r1[144];
              r1[144] = (v309_data + (v281_data * v97_data));
              float v311_data = r0[16];
              float v314_data = r1[16];
              r1[16] = (v314_data + (v311_data * v72_data));
              float v319_data = r1[42];
              r1[42] = (v319_data + (v311_data * v77_data));
              float v324_data = r1[68];
              r1[68] = (v324_data + (v311_data * v82_data));
              float v329_data = r1[94];
              r1[94] = (v329_data + (v311_data * v87_data));
              float v334_data = r1[120];
              r1[120] = (v334_data + (v311_data * v92_data));
              float v339_data = r1[146];
              r1[146] = (v339_data + (v311_data * v97_data));
              float v341_data = r0[18];
              float v344_data = r1[18];
              r1[18] = (v344_data + (v341_data * v72_data));
              float v349_data = r1[44];
              r1[44] = (v349_data + (v341_data * v77_data));
              float v354_data = r1[70];
              r1[70] = (v354_data + (v341_data * v82_data));
              float v359_data = r1[96];
              r1[96] = (v359_data + (v341_data * v87_data));
              float v364_data = r1[122];
              r1[122] = (v364_data + (v341_data * v92_data));
              float v369_data = r1[148];
              r1[148] = (v369_data + (v341_data * v97_data));
              float v371_data = r0[20];
              float v374_data = r1[20];
              r1[20] = (v374_data + (v371_data * v72_data));
              float v379_data = r1[46];
              r1[46] = (v379_data + (v371_data * v77_data));
              float v384_data = r1[72];
              r1[72] = (v384_data + (v371_data * v82_data));
              float v389_data = r1[98];
              r1[98] = (v389_data + (v371_data * v87_data));
              float v394_data = r1[124];
              r1[124] = (v394_data + (v371_data * v92_data));
              float v399_data = r1[150];
              r1[150] = (v399_data + (v371_data * v97_data));
              float v401_data = r0[22];
              float v404_data = r1[22];
              r1[22] = (v404_data + (v401_data * v72_data));
              float v409_data = r1[48];
              r1[48] = (v409_data + (v401_data * v77_data));
              float v414_data = r1[74];
              r1[74] = (v414_data + (v401_data * v82_data));
              float v419_data = r1[100];
              r1[100] = (v419_data + (v401_data * v87_data));
              float v424_data = r1[126];
              r1[126] = (v424_data + (v401_data * v92_data));
              float v429_data = r1[152];
              r1[152] = (v429_data + (v401_data * v97_data));
              float v431_data = r0[24];
              float v434_data = r1[24];
              r1[24] = (v434_data + (v431_data * v72_data));
              float v439_data = r1[50];
              r1[50] = (v439_data + (v431_data * v77_data));
              float v444_data = r1[76];
              r1[76] = (v444_data + (v431_data * v82_data));
              float v449_data = r1[102];
              r1[102] = (v449_data + (v431_data * v87_data));
              float v454_data = r1[128];
              r1[128] = (v454_data + (v431_data * v92_data));
              float v459_data = r1[154];
              r1[154] = (v459_data + (v431_data * v97_data));
              float v461_data = r0[1];
              float v464_data = r1[1];
              r1[1] = (v464_data + (v461_data * v72_data));
              float v469_data = r1[27];
              r1[27] = (v469_data + (v461_data * v77_data));
              float v474_data = r1[53];
              r1[53] = (v474_data + (v461_data * v82_data));
              float v479_data = r1[79];
              r1[79] = (v479_data + (v461_data * v87_data));
              float v484_data = r1[105];
              r1[105] = (v484_data + (v461_data * v92_data));
              float v489_data = r1[131];
              r1[131] = (v489_data + (v461_data * v97_data));
              float v491_data = r0[3];
              float v494_data = r1[3];
              r1[3] = (v494_data + (v491_data * v72_data));
              float v499_data = r1[29];
              r1[29] = (v499_data + (v491_data * v77_data));
              float v504_data = r1[55];
              r1[55] = (v504_data + (v491_data * v82_data));
              float v509_data = r1[81];
              r1[81] = (v509_data + (v491_data * v87_data));
              float v514_data = r1[107];
              r1[107] = (v514_data + (v491_data * v92_data));
              float v519_data = r1[133];
              r1[133] = (v519_data + (v491_data * v97_data));
              float v521_data = r0[5];
              float v524_data = r1[5];
              r1[5] = (v524_data + (v521_data * v72_data));
              float v529_data = r1[31];
              r1[31] = (v529_data + (v521_data * v77_data));
              float v534_data = r1[57];
              r1[57] = (v534_data + (v521_data * v82_data));
              float v539_data = r1[83];
              r1[83] = (v539_data + (v521_data * v87_data));
              float v544_data = r1[109];
              r1[109] = (v544_data + (v521_data * v92_data));
              float v549_data = r1[135];
              r1[135] = (v549_data + (v521_data * v97_data));
              float v551_data = r0[7];
              float v554_data = r1[7];
              r1[7] = (v554_data + (v551_data * v72_data));
              float v559_data = r1[33];
              r1[33] = (v559_data + (v551_data * v77_data));
              float v564_data = r1[59];
              r1[59] = (v564_data + (v551_data * v82_data));
              float v569_data = r1[85];
              r1[85] = (v569_data + (v551_data * v87_data));
              float v574_data = r1[111];
              r1[111] = (v574_data + (v551_data * v92_data));
              float v579_data = r1[137];
              r1[137] = (v579_data + (v551_data * v97_data));
              float v581_data = r0[9];
              float v584_data = r1[9];
              r1[9] = (v584_data + (v581_data * v72_data));
              float v589_data = r1[35];
              r1[35] = (v589_data + (v581_data * v77_data));
              float v594_data = r1[61];
              r1[61] = (v594_data + (v581_data * v82_data));
              float v599_data = r1[87];
              r1[87] = (v599_data + (v581_data * v87_data));
              float v604_data = r1[113];
              r1[113] = (v604_data + (v581_data * v92_data));
              float v609_data = r1[139];
              r1[139] = (v609_data + (v581_data * v97_data));
              float v611_data = r0[11];
              float v614_data = r1[11];
              r1[11] = (v614_data + (v611_data * v72_data));
              float v619_data = r1[37];
              r1[37] = (v619_data + (v611_data * v77_data));
              float v624_data = r1[63];
              r1[63] = (v624_data + (v611_data * v82_data));
              float v629_data = r1[89];
              r1[89] = (v629_data + (v611_data * v87_data));
              float v634_data = r1[115];
              r1[115] = (v634_data + (v611_data * v92_data));
              float v639_data = r1[141];
              r1[141] = (v639_data + (v611_data * v97_data));
              float v641_data = r0[13];
              float v644_data = r1[13];
              r1[13] = (v644_data + (v641_data * v72_data));
              float v649_data = r1[39];
              r1[39] = (v649_data + (v641_data * v77_data));
              float v654_data = r1[65];
              r1[65] = (v654_data + (v641_data * v82_data));
              float v659_data = r1[91];
              r1[91] = (v659_data + (v641_data * v87_data));
              float v664_data = r1[117];
              r1[117] = (v664_data + (v641_data * v92_data));
              float v669_data = r1[143];
              r1[143] = (v669_data + (v641_data * v97_data));
              float v671_data = r0[15];
              float v674_data = r1[15];
              r1[15] = (v674_data + (v671_data * v72_data));
              float v679_data = r1[41];
              r1[41] = (v679_data + (v671_data * v77_data));
              float v684_data = r1[67];
              r1[67] = (v684_data + (v671_data * v82_data));
              float v689_data = r1[93];
              r1[93] = (v689_data + (v671_data * v87_data));
              float v694_data = r1[119];
              r1[119] = (v694_data + (v671_data * v92_data));
              float v699_data = r1[145];
              r1[145] = (v699_data + (v671_data * v97_data));
              float v701_data = r0[17];
              float v704_data = r1[17];
              r1[17] = (v704_data + (v701_data * v72_data));
              float v709_data = r1[43];
              r1[43] = (v709_data + (v701_data * v77_data));
              float v714_data = r1[69];
              r1[69] = (v714_data + (v701_data * v82_data));
              float v719_data = r1[95];
              r1[95] = (v719_data + (v701_data * v87_data));
              float v724_data = r1[121];
              r1[121] = (v724_data + (v701_data * v92_data));
              float v729_data = r1[147];
              r1[147] = (v729_data + (v701_data * v97_data));
              float v731_data = r0[19];
              float v734_data = r1[19];
              r1[19] = (v734_data + (v731_data * v72_data));
              float v739_data = r1[45];
              r1[45] = (v739_data + (v731_data * v77_data));
              float v744_data = r1[71];
              r1[71] = (v744_data + (v731_data * v82_data));
              float v749_data = r1[97];
              r1[97] = (v749_data + (v731_data * v87_data));
              float v754_data = r1[123];
              r1[123] = (v754_data + (v731_data * v92_data));
              float v759_data = r1[149];
              r1[149] = (v759_data + (v731_data * v97_data));
              float v761_data = r0[21];
              float v764_data = r1[21];
              r1[21] = (v764_data + (v761_data * v72_data));
              float v769_data = r1[47];
              r1[47] = (v769_data + (v761_data * v77_data));
              float v774_data = r1[73];
              r1[73] = (v774_data + (v761_data * v82_data));
              float v779_data = r1[99];
              r1[99] = (v779_data + (v761_data * v87_data));
              float v784_data = r1[125];
              r1[125] = (v784_data + (v761_data * v92_data));
              float v789_data = r1[151];
              r1[151] = (v789_data + (v761_data * v97_data));
              float v791_data = r0[23];
              float v794_data = r1[23];
              r1[23] = (v794_data + (v791_data * v72_data));
              float v799_data = r1[49];
              r1[49] = (v799_data + (v791_data * v77_data));
              float v804_data = r1[75];
              r1[75] = (v804_data + (v791_data * v82_data));
              float v809_data = r1[101];
              r1[101] = (v809_data + (v791_data * v87_data));
              float v814_data = r1[127];
              r1[127] = (v814_data + (v791_data * v92_data));
              float v819_data = r1[153];
              r1[153] = (v819_data + (v791_data * v97_data));
              float v821_data = r0[25];
              float v824_data = r1[25];
              r1[25] = (v824_data + (v821_data * v72_data));
              float v829_data = r1[51];
              r1[51] = (v829_data + (v821_data * v77_data));
              float v834_data = r1[77];
              r1[77] = (v834_data + (v821_data * v82_data));
              float v839_data = r1[103];
              r1[103] = (v839_data + (v821_data * v87_data));
              float v844_data = r1[129];
              r1[129] = (v844_data + (v821_data * v92_data));
              float v849_data = r1[155];
              r1[155] = (v849_data + (v821_data * v97_data));
              // wait(r2 = load{g>r}(glb_m2););
              float r3[12]{};
              // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(20, 35), (0, 1), (0, 6)] []
              float ir3[12]{};
              if (v14_lead >= 20) {
                float v857_data = r1[24];
                float v858_data = ir3[0];
                ir3[0] = (v858_data + v857_data);
                float v860_data = r1[50];
                float v861_data = ir3[2];
                ir3[2] = (v861_data + v860_data);
                float v863_data = r1[76];
                float v864_data = ir3[4];
                ir3[4] = (v864_data + v863_data);
                float v866_data = r1[102];
                float v867_data = ir3[6];
                ir3[6] = (v867_data + v866_data);
                float v869_data = r1[128];
                float v870_data = ir3[8];
                ir3[8] = (v870_data + v869_data);
                float v872_data = r1[154];
                float v873_data = ir3[10];
                ir3[10] = (v873_data + v872_data);
              }
              if (v14_lead < 3) {
                float v876_data = r1[25];
                float v877_data = ir3[1];
                ir3[1] = (v877_data + v876_data);
                float v879_data = r1[51];
                float v880_data = ir3[3];
                ir3[3] = (v880_data + v879_data);
                float v882_data = r1[77];
                float v883_data = ir3[5];
                ir3[5] = (v883_data + v882_data);
                float v885_data = r1[103];
                float v886_data = ir3[7];
                ir3[7] = (v886_data + v885_data);
                float v888_data = r1[129];
                float v889_data = ir3[9];
                ir3[9] = (v889_data + v888_data);
                float v891_data = r1[155];
                float v892_data = ir3[11];
                ir3[11] = (v892_data + v891_data);
              }
              if (v14_lead >= 20) {
                #pragma unroll
                for (int32_t v898_n1 = 0; v898_n1 < 1; ++v898_n1) {
                  int32_t v900_a = v898_n1 * 2;
                  #pragma unroll
                  for (int32_t v899_n2 = 0; v899_n2 < 6; ++v899_n2) {
                    int32_t v903_a = v900_a + (v899_n2 * 2);
                    float v904_data = ir3[v903_a];
                    float v909_data = r2[v903_a];
                    r3[v903_a] = (v909_data + v904_data);
                  }
                }
              }
              if (v14_lead < 3) {
                #pragma unroll
                for (int32_t v916_n1 = 0; v916_n1 < 1; ++v916_n1) {
                  int32_t v920_a = 1 + (v916_n1 * 2);
                  #pragma unroll
                  for (int32_t v917_n2 = 0; v917_n2 < 6; ++v917_n2) {
                    int32_t v919_a = v917_n2 * 2;
                    float v922_data = ir3[(v920_a + v919_a)];
                    float v927_data = r2[(v920_a + v919_a)];
                    r3[(v920_a + v919_a)] = (v927_data + v922_data);
                  }
                }
              }
              // glb_m2 = store{r>g}(r3);
              if (v14_lead >= 20) {
                #pragma unroll
                for (int32_t v937_i1 = 0; v937_i1 < 1; ++v937_i1) {
                  int32_t v939_a = v937_i1 * 2;
                  int32_t v952_a = v14_lead + ((v937_i1 + 12) * 64);
                  #pragma unroll
                  for (int32_t v938_i2 = 0; v938_i2 < 6; ++v938_i2) {
                    float v943_data = r3[(v939_a + (v938_i2 * 2))];
                    glb_m2[(v952_a + (v938_i2 * 832))] = v943_data;
                  }
                }
              }
              if (v14_lead < 3) {
                int32_t v966_lead = v14_lead + 32_i32;
                #pragma unroll
                for (int32_t v955_i1 = 0; v955_i1 < 1; ++v955_i1) {
                  int32_t v959_a = 1 + (v955_i1 * 2);
                  int32_t v970_a = v966_lead + ((v955_i1 + 12) * 64);
                  #pragma unroll
                  for (int32_t v956_i2 = 0; v956_i2 < 6; ++v956_i2) {
                    float v961_data = r3[(v959_a + (v956_i2 * 2))];
                    glb_m2[(v970_a + (v956_i2 * 832))] = v961_data;
                  }
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

