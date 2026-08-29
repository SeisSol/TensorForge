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
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2304, cgh); {
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
          float* localShrMem0 = &totalShrMem[144 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[128];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 512 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[8]{};
              sycl::group_barrier(item.get_sub_group());
              // r0 = +(glb_m1 * s0) + None
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir0[8]{};
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 12) {
                float v17_data = glb_m1[(v8_lead + 4)];
                float v18_data = s0[0];
                float v20_data = ir0[0];
                ir0[0] = (v20_data + (v17_data * v18_data));
                float v29_data = glb_m1[(v8_lead + 4)];
                float v30_data = s0[16];
                float v32_data = ir0[1];
                ir0[1] = (v32_data + (v29_data * v30_data));
                float v41_data = glb_m1[(v8_lead + 4)];
                float v42_data = s0[33];
                float v44_data = ir0[2];
                ir0[2] = (v44_data + (v41_data * v42_data));
                float v53_data = glb_m1[(v8_lead + 4)];
                float v54_data = s0[49];
                float v56_data = ir0[3];
                ir0[3] = (v56_data + (v53_data * v54_data));
                float v65_data = glb_m1[(v8_lead + 4)];
                float v66_data = s0[66];
                float v68_data = ir0[4];
                ir0[4] = (v68_data + (v65_data * v66_data));
                float v77_data = glb_m1[(v8_lead + 4)];
                float v78_data = s0[82];
                float v80_data = ir0[5];
                ir0[5] = (v80_data + (v77_data * v78_data));
                float v89_data = glb_m1[(v8_lead + 4)];
                float v90_data = s0[99];
                float v92_data = ir0[6];
                ir0[6] = (v92_data + (v89_data * v90_data));
                float v101_data = glb_m1[(v8_lead + 4)];
                float v102_data = s0[115];
                float v104_data = ir0[7];
                ir0[7] = (v104_data + (v101_data * v102_data));
              }
              if (v8_lead < 12) {
                float v117_data = glb_m1[((v8_lead + 4) + 32)];
                float v118_data = s0[1];
                float v120_data = ir0[0];
                ir0[0] = (v120_data + (v117_data * v118_data));
                float v129_data = glb_m1[((v8_lead + 4) + 32)];
                float v130_data = s0[17];
                float v132_data = ir0[1];
                ir0[1] = (v132_data + (v129_data * v130_data));
                float v141_data = glb_m1[((v8_lead + 4) + 32)];
                float v142_data = s0[32];
                float v144_data = ir0[2];
                ir0[2] = (v144_data + (v141_data * v142_data));
                float v153_data = glb_m1[((v8_lead + 4) + 32)];
                float v154_data = s0[48];
                float v156_data = ir0[3];
                ir0[3] = (v156_data + (v153_data * v154_data));
                float v165_data = glb_m1[((v8_lead + 4) + 32)];
                float v166_data = s0[67];
                float v168_data = ir0[4];
                ir0[4] = (v168_data + (v165_data * v166_data));
                float v177_data = glb_m1[((v8_lead + 4) + 32)];
                float v178_data = s0[83];
                float v180_data = ir0[5];
                ir0[5] = (v180_data + (v177_data * v178_data));
                float v189_data = glb_m1[((v8_lead + 4) + 32)];
                float v190_data = s0[98];
                float v192_data = ir0[6];
                ir0[6] = (v192_data + (v189_data * v190_data));
                float v201_data = glb_m1[((v8_lead + 4) + 32)];
                float v202_data = s0[114];
                float v204_data = ir0[7];
                ir0[7] = (v204_data + (v201_data * v202_data));
              }
              if (v8_lead < 12) {
                float v217_data = glb_m1[((v8_lead + 4) + 64)];
                float v218_data = s0[2];
                float v220_data = ir0[0];
                ir0[0] = (v220_data + (v217_data * v218_data));
                float v229_data = glb_m1[((v8_lead + 4) + 64)];
                float v230_data = s0[18];
                float v232_data = ir0[1];
                ir0[1] = (v232_data + (v229_data * v230_data));
                float v241_data = glb_m1[((v8_lead + 4) + 64)];
                float v242_data = s0[35];
                float v244_data = ir0[2];
                ir0[2] = (v244_data + (v241_data * v242_data));
                float v253_data = glb_m1[((v8_lead + 4) + 64)];
                float v254_data = s0[51];
                float v256_data = ir0[3];
                ir0[3] = (v256_data + (v253_data * v254_data));
                float v265_data = glb_m1[((v8_lead + 4) + 64)];
                float v266_data = s0[64];
                float v268_data = ir0[4];
                ir0[4] = (v268_data + (v265_data * v266_data));
                float v277_data = glb_m1[((v8_lead + 4) + 64)];
                float v278_data = s0[80];
                float v280_data = ir0[5];
                ir0[5] = (v280_data + (v277_data * v278_data));
                float v289_data = glb_m1[((v8_lead + 4) + 64)];
                float v290_data = s0[97];
                float v292_data = ir0[6];
                ir0[6] = (v292_data + (v289_data * v290_data));
                float v301_data = glb_m1[((v8_lead + 4) + 64)];
                float v302_data = s0[113];
                float v304_data = ir0[7];
                ir0[7] = (v304_data + (v301_data * v302_data));
              }
              if (v8_lead < 12) {
                float v317_data = glb_m1[((v8_lead + 4) + 96)];
                float v318_data = s0[3];
                float v320_data = ir0[0];
                ir0[0] = (v320_data + (v317_data * v318_data));
                float v329_data = glb_m1[((v8_lead + 4) + 96)];
                float v330_data = s0[19];
                float v332_data = ir0[1];
                ir0[1] = (v332_data + (v329_data * v330_data));
                float v341_data = glb_m1[((v8_lead + 4) + 96)];
                float v342_data = s0[34];
                float v344_data = ir0[2];
                ir0[2] = (v344_data + (v341_data * v342_data));
                float v353_data = glb_m1[((v8_lead + 4) + 96)];
                float v354_data = s0[50];
                float v356_data = ir0[3];
                ir0[3] = (v356_data + (v353_data * v354_data));
                float v365_data = glb_m1[((v8_lead + 4) + 96)];
                float v366_data = s0[65];
                float v368_data = ir0[4];
                ir0[4] = (v368_data + (v365_data * v366_data));
                float v377_data = glb_m1[((v8_lead + 4) + 96)];
                float v378_data = s0[81];
                float v380_data = ir0[5];
                ir0[5] = (v380_data + (v377_data * v378_data));
                float v389_data = glb_m1[((v8_lead + 4) + 96)];
                float v390_data = s0[96];
                float v392_data = ir0[6];
                ir0[6] = (v392_data + (v389_data * v390_data));
                float v401_data = glb_m1[((v8_lead + 4) + 96)];
                float v402_data = s0[112];
                float v404_data = ir0[7];
                ir0[7] = (v404_data + (v401_data * v402_data));
              }
              if (v8_lead < 12) {
                float v417_data = glb_m1[((v8_lead + 4) + 128)];
                float v418_data = s0[4];
                float v420_data = ir0[0];
                ir0[0] = (v420_data + (v417_data * v418_data));
                float v429_data = glb_m1[((v8_lead + 4) + 128)];
                float v430_data = s0[20];
                float v432_data = ir0[1];
                ir0[1] = (v432_data + (v429_data * v430_data));
                float v441_data = glb_m1[((v8_lead + 4) + 128)];
                float v442_data = s0[37];
                float v444_data = ir0[2];
                ir0[2] = (v444_data + (v441_data * v442_data));
                float v453_data = glb_m1[((v8_lead + 4) + 128)];
                float v454_data = s0[53];
                float v456_data = ir0[3];
                ir0[3] = (v456_data + (v453_data * v454_data));
                float v465_data = glb_m1[((v8_lead + 4) + 128)];
                float v466_data = s0[70];
                float v468_data = ir0[4];
                ir0[4] = (v468_data + (v465_data * v466_data));
                float v477_data = glb_m1[((v8_lead + 4) + 128)];
                float v478_data = s0[86];
                float v480_data = ir0[5];
                ir0[5] = (v480_data + (v477_data * v478_data));
                float v489_data = glb_m1[((v8_lead + 4) + 128)];
                float v490_data = s0[103];
                float v492_data = ir0[6];
                ir0[6] = (v492_data + (v489_data * v490_data));
                float v501_data = glb_m1[((v8_lead + 4) + 128)];
                float v502_data = s0[119];
                float v504_data = ir0[7];
                ir0[7] = (v504_data + (v501_data * v502_data));
              }
              if (v8_lead < 12) {
                float v517_data = glb_m1[((v8_lead + 4) + 160)];
                float v518_data = s0[5];
                float v520_data = ir0[0];
                ir0[0] = (v520_data + (v517_data * v518_data));
                float v529_data = glb_m1[((v8_lead + 4) + 160)];
                float v530_data = s0[21];
                float v532_data = ir0[1];
                ir0[1] = (v532_data + (v529_data * v530_data));
                float v541_data = glb_m1[((v8_lead + 4) + 160)];
                float v542_data = s0[36];
                float v544_data = ir0[2];
                ir0[2] = (v544_data + (v541_data * v542_data));
                float v553_data = glb_m1[((v8_lead + 4) + 160)];
                float v554_data = s0[52];
                float v556_data = ir0[3];
                ir0[3] = (v556_data + (v553_data * v554_data));
                float v565_data = glb_m1[((v8_lead + 4) + 160)];
                float v566_data = s0[71];
                float v568_data = ir0[4];
                ir0[4] = (v568_data + (v565_data * v566_data));
                float v577_data = glb_m1[((v8_lead + 4) + 160)];
                float v578_data = s0[87];
                float v580_data = ir0[5];
                ir0[5] = (v580_data + (v577_data * v578_data));
                float v589_data = glb_m1[((v8_lead + 4) + 160)];
                float v590_data = s0[102];
                float v592_data = ir0[6];
                ir0[6] = (v592_data + (v589_data * v590_data));
                float v601_data = glb_m1[((v8_lead + 4) + 160)];
                float v602_data = s0[118];
                float v604_data = ir0[7];
                ir0[7] = (v604_data + (v601_data * v602_data));
              }
              if (v8_lead < 12) {
                float v617_data = glb_m1[((v8_lead + 4) + 192)];
                float v618_data = s0[6];
                float v620_data = ir0[0];
                ir0[0] = (v620_data + (v617_data * v618_data));
                float v629_data = glb_m1[((v8_lead + 4) + 192)];
                float v630_data = s0[22];
                float v632_data = ir0[1];
                ir0[1] = (v632_data + (v629_data * v630_data));
                float v641_data = glb_m1[((v8_lead + 4) + 192)];
                float v642_data = s0[39];
                float v644_data = ir0[2];
                ir0[2] = (v644_data + (v641_data * v642_data));
                float v653_data = glb_m1[((v8_lead + 4) + 192)];
                float v654_data = s0[55];
                float v656_data = ir0[3];
                ir0[3] = (v656_data + (v653_data * v654_data));
                float v665_data = glb_m1[((v8_lead + 4) + 192)];
                float v666_data = s0[68];
                float v668_data = ir0[4];
                ir0[4] = (v668_data + (v665_data * v666_data));
                float v677_data = glb_m1[((v8_lead + 4) + 192)];
                float v678_data = s0[84];
                float v680_data = ir0[5];
                ir0[5] = (v680_data + (v677_data * v678_data));
                float v689_data = glb_m1[((v8_lead + 4) + 192)];
                float v690_data = s0[101];
                float v692_data = ir0[6];
                ir0[6] = (v692_data + (v689_data * v690_data));
                float v701_data = glb_m1[((v8_lead + 4) + 192)];
                float v702_data = s0[117];
                float v704_data = ir0[7];
                ir0[7] = (v704_data + (v701_data * v702_data));
              }
              if (v8_lead < 12) {
                float v717_data = glb_m1[((v8_lead + 4) + 224)];
                float v718_data = s0[7];
                float v720_data = ir0[0];
                ir0[0] = (v720_data + (v717_data * v718_data));
                float v729_data = glb_m1[((v8_lead + 4) + 224)];
                float v730_data = s0[23];
                float v732_data = ir0[1];
                ir0[1] = (v732_data + (v729_data * v730_data));
                float v741_data = glb_m1[((v8_lead + 4) + 224)];
                float v742_data = s0[38];
                float v744_data = ir0[2];
                ir0[2] = (v744_data + (v741_data * v742_data));
                float v753_data = glb_m1[((v8_lead + 4) + 224)];
                float v754_data = s0[54];
                float v756_data = ir0[3];
                ir0[3] = (v756_data + (v753_data * v754_data));
                float v765_data = glb_m1[((v8_lead + 4) + 224)];
                float v766_data = s0[69];
                float v768_data = ir0[4];
                ir0[4] = (v768_data + (v765_data * v766_data));
                float v777_data = glb_m1[((v8_lead + 4) + 224)];
                float v778_data = s0[85];
                float v780_data = ir0[5];
                ir0[5] = (v780_data + (v777_data * v778_data));
                float v789_data = glb_m1[((v8_lead + 4) + 224)];
                float v790_data = s0[100];
                float v792_data = ir0[6];
                ir0[6] = (v792_data + (v789_data * v790_data));
                float v801_data = glb_m1[((v8_lead + 4) + 224)];
                float v802_data = s0[116];
                float v804_data = ir0[7];
                ir0[7] = (v804_data + (v801_data * v802_data));
              }
              if (v8_lead < 12) {
                float v817_data = glb_m1[((v8_lead + 4) + 256)];
                float v818_data = s0[8];
                float v820_data = ir0[0];
                ir0[0] = (v820_data + (v817_data * v818_data));
                float v829_data = glb_m1[((v8_lead + 4) + 256)];
                float v830_data = s0[24];
                float v832_data = ir0[1];
                ir0[1] = (v832_data + (v829_data * v830_data));
                float v841_data = glb_m1[((v8_lead + 4) + 256)];
                float v842_data = s0[41];
                float v844_data = ir0[2];
                ir0[2] = (v844_data + (v841_data * v842_data));
                float v853_data = glb_m1[((v8_lead + 4) + 256)];
                float v854_data = s0[57];
                float v856_data = ir0[3];
                ir0[3] = (v856_data + (v853_data * v854_data));
                float v865_data = glb_m1[((v8_lead + 4) + 256)];
                float v866_data = s0[74];
                float v868_data = ir0[4];
                ir0[4] = (v868_data + (v865_data * v866_data));
                float v877_data = glb_m1[((v8_lead + 4) + 256)];
                float v878_data = s0[90];
                float v880_data = ir0[5];
                ir0[5] = (v880_data + (v877_data * v878_data));
                float v889_data = glb_m1[((v8_lead + 4) + 256)];
                float v890_data = s0[107];
                float v892_data = ir0[6];
                ir0[6] = (v892_data + (v889_data * v890_data));
                float v901_data = glb_m1[((v8_lead + 4) + 256)];
                float v902_data = s0[123];
                float v904_data = ir0[7];
                ir0[7] = (v904_data + (v901_data * v902_data));
              }
              if (v8_lead < 12) {
                float v917_data = glb_m1[((v8_lead + 4) + 288)];
                float v918_data = s0[9];
                float v920_data = ir0[0];
                ir0[0] = (v920_data + (v917_data * v918_data));
                float v929_data = glb_m1[((v8_lead + 4) + 288)];
                float v930_data = s0[25];
                float v932_data = ir0[1];
                ir0[1] = (v932_data + (v929_data * v930_data));
                float v941_data = glb_m1[((v8_lead + 4) + 288)];
                float v942_data = s0[40];
                float v944_data = ir0[2];
                ir0[2] = (v944_data + (v941_data * v942_data));
                float v953_data = glb_m1[((v8_lead + 4) + 288)];
                float v954_data = s0[56];
                float v956_data = ir0[3];
                ir0[3] = (v956_data + (v953_data * v954_data));
                float v965_data = glb_m1[((v8_lead + 4) + 288)];
                float v966_data = s0[75];
                float v968_data = ir0[4];
                ir0[4] = (v968_data + (v965_data * v966_data));
                float v977_data = glb_m1[((v8_lead + 4) + 288)];
                float v978_data = s0[91];
                float v980_data = ir0[5];
                ir0[5] = (v980_data + (v977_data * v978_data));
                float v989_data = glb_m1[((v8_lead + 4) + 288)];
                float v990_data = s0[106];
                float v992_data = ir0[6];
                ir0[6] = (v992_data + (v989_data * v990_data));
                float v1001_data = glb_m1[((v8_lead + 4) + 288)];
                float v1002_data = s0[122];
                float v1004_data = ir0[7];
                ir0[7] = (v1004_data + (v1001_data * v1002_data));
              }
              if (v8_lead < 12) {
                float v1017_data = glb_m1[((v8_lead + 4) + 320)];
                float v1018_data = s0[10];
                float v1020_data = ir0[0];
                ir0[0] = (v1020_data + (v1017_data * v1018_data));
                float v1029_data = glb_m1[((v8_lead + 4) + 320)];
                float v1030_data = s0[26];
                float v1032_data = ir0[1];
                ir0[1] = (v1032_data + (v1029_data * v1030_data));
                float v1041_data = glb_m1[((v8_lead + 4) + 320)];
                float v1042_data = s0[43];
                float v1044_data = ir0[2];
                ir0[2] = (v1044_data + (v1041_data * v1042_data));
                float v1053_data = glb_m1[((v8_lead + 4) + 320)];
                float v1054_data = s0[59];
                float v1056_data = ir0[3];
                ir0[3] = (v1056_data + (v1053_data * v1054_data));
                float v1065_data = glb_m1[((v8_lead + 4) + 320)];
                float v1066_data = s0[72];
                float v1068_data = ir0[4];
                ir0[4] = (v1068_data + (v1065_data * v1066_data));
                float v1077_data = glb_m1[((v8_lead + 4) + 320)];
                float v1078_data = s0[88];
                float v1080_data = ir0[5];
                ir0[5] = (v1080_data + (v1077_data * v1078_data));
                float v1089_data = glb_m1[((v8_lead + 4) + 320)];
                float v1090_data = s0[105];
                float v1092_data = ir0[6];
                ir0[6] = (v1092_data + (v1089_data * v1090_data));
                float v1101_data = glb_m1[((v8_lead + 4) + 320)];
                float v1102_data = s0[121];
                float v1104_data = ir0[7];
                ir0[7] = (v1104_data + (v1101_data * v1102_data));
              }
              if (v8_lead < 12) {
                float v1117_data = glb_m1[((v8_lead + 4) + 352)];
                float v1118_data = s0[11];
                float v1120_data = ir0[0];
                ir0[0] = (v1120_data + (v1117_data * v1118_data));
                float v1129_data = glb_m1[((v8_lead + 4) + 352)];
                float v1130_data = s0[27];
                float v1132_data = ir0[1];
                ir0[1] = (v1132_data + (v1129_data * v1130_data));
                float v1141_data = glb_m1[((v8_lead + 4) + 352)];
                float v1142_data = s0[42];
                float v1144_data = ir0[2];
                ir0[2] = (v1144_data + (v1141_data * v1142_data));
                float v1153_data = glb_m1[((v8_lead + 4) + 352)];
                float v1154_data = s0[58];
                float v1156_data = ir0[3];
                ir0[3] = (v1156_data + (v1153_data * v1154_data));
                float v1165_data = glb_m1[((v8_lead + 4) + 352)];
                float v1166_data = s0[73];
                float v1168_data = ir0[4];
                ir0[4] = (v1168_data + (v1165_data * v1166_data));
                float v1177_data = glb_m1[((v8_lead + 4) + 352)];
                float v1178_data = s0[89];
                float v1180_data = ir0[5];
                ir0[5] = (v1180_data + (v1177_data * v1178_data));
                float v1189_data = glb_m1[((v8_lead + 4) + 352)];
                float v1190_data = s0[104];
                float v1192_data = ir0[6];
                ir0[6] = (v1192_data + (v1189_data * v1190_data));
                float v1201_data = glb_m1[((v8_lead + 4) + 352)];
                float v1202_data = s0[120];
                float v1204_data = ir0[7];
                ir0[7] = (v1204_data + (v1201_data * v1202_data));
              }
              if (v8_lead < 12) {
                float v1217_data = glb_m1[((v8_lead + 4) + 384)];
                float v1218_data = s0[12];
                float v1220_data = ir0[0];
                ir0[0] = (v1220_data + (v1217_data * v1218_data));
                float v1229_data = glb_m1[((v8_lead + 4) + 384)];
                float v1230_data = s0[28];
                float v1232_data = ir0[1];
                ir0[1] = (v1232_data + (v1229_data * v1230_data));
                float v1241_data = glb_m1[((v8_lead + 4) + 384)];
                float v1242_data = s0[45];
                float v1244_data = ir0[2];
                ir0[2] = (v1244_data + (v1241_data * v1242_data));
                float v1253_data = glb_m1[((v8_lead + 4) + 384)];
                float v1254_data = s0[61];
                float v1256_data = ir0[3];
                ir0[3] = (v1256_data + (v1253_data * v1254_data));
                float v1265_data = glb_m1[((v8_lead + 4) + 384)];
                float v1266_data = s0[78];
                float v1268_data = ir0[4];
                ir0[4] = (v1268_data + (v1265_data * v1266_data));
                float v1277_data = glb_m1[((v8_lead + 4) + 384)];
                float v1278_data = s0[94];
                float v1280_data = ir0[5];
                ir0[5] = (v1280_data + (v1277_data * v1278_data));
                float v1289_data = glb_m1[((v8_lead + 4) + 384)];
                float v1290_data = s0[111];
                float v1292_data = ir0[6];
                ir0[6] = (v1292_data + (v1289_data * v1290_data));
                float v1301_data = glb_m1[((v8_lead + 4) + 384)];
                float v1302_data = s0[127];
                float v1304_data = ir0[7];
                ir0[7] = (v1304_data + (v1301_data * v1302_data));
              }
              if (v8_lead < 12) {
                float v1317_data = glb_m1[((v8_lead + 4) + 416)];
                float v1318_data = s0[13];
                float v1320_data = ir0[0];
                ir0[0] = (v1320_data + (v1317_data * v1318_data));
                float v1329_data = glb_m1[((v8_lead + 4) + 416)];
                float v1330_data = s0[29];
                float v1332_data = ir0[1];
                ir0[1] = (v1332_data + (v1329_data * v1330_data));
                float v1341_data = glb_m1[((v8_lead + 4) + 416)];
                float v1342_data = s0[44];
                float v1344_data = ir0[2];
                ir0[2] = (v1344_data + (v1341_data * v1342_data));
                float v1353_data = glb_m1[((v8_lead + 4) + 416)];
                float v1354_data = s0[60];
                float v1356_data = ir0[3];
                ir0[3] = (v1356_data + (v1353_data * v1354_data));
                float v1365_data = glb_m1[((v8_lead + 4) + 416)];
                float v1366_data = s0[79];
                float v1368_data = ir0[4];
                ir0[4] = (v1368_data + (v1365_data * v1366_data));
                float v1377_data = glb_m1[((v8_lead + 4) + 416)];
                float v1378_data = s0[95];
                float v1380_data = ir0[5];
                ir0[5] = (v1380_data + (v1377_data * v1378_data));
                float v1389_data = glb_m1[((v8_lead + 4) + 416)];
                float v1390_data = s0[110];
                float v1392_data = ir0[6];
                ir0[6] = (v1392_data + (v1389_data * v1390_data));
                float v1401_data = glb_m1[((v8_lead + 4) + 416)];
                float v1402_data = s0[126];
                float v1404_data = ir0[7];
                ir0[7] = (v1404_data + (v1401_data * v1402_data));
              }
              if (v8_lead < 12) {
                float v1417_data = glb_m1[((v8_lead + 4) + 448)];
                float v1418_data = s0[14];
                float v1420_data = ir0[0];
                ir0[0] = (v1420_data + (v1417_data * v1418_data));
                float v1429_data = glb_m1[((v8_lead + 4) + 448)];
                float v1430_data = s0[30];
                float v1432_data = ir0[1];
                ir0[1] = (v1432_data + (v1429_data * v1430_data));
                float v1441_data = glb_m1[((v8_lead + 4) + 448)];
                float v1442_data = s0[47];
                float v1444_data = ir0[2];
                ir0[2] = (v1444_data + (v1441_data * v1442_data));
                float v1453_data = glb_m1[((v8_lead + 4) + 448)];
                float v1454_data = s0[63];
                float v1456_data = ir0[3];
                ir0[3] = (v1456_data + (v1453_data * v1454_data));
                float v1465_data = glb_m1[((v8_lead + 4) + 448)];
                float v1466_data = s0[76];
                float v1468_data = ir0[4];
                ir0[4] = (v1468_data + (v1465_data * v1466_data));
                float v1477_data = glb_m1[((v8_lead + 4) + 448)];
                float v1478_data = s0[92];
                float v1480_data = ir0[5];
                ir0[5] = (v1480_data + (v1477_data * v1478_data));
                float v1489_data = glb_m1[((v8_lead + 4) + 448)];
                float v1490_data = s0[109];
                float v1492_data = ir0[6];
                ir0[6] = (v1492_data + (v1489_data * v1490_data));
                float v1501_data = glb_m1[((v8_lead + 4) + 448)];
                float v1502_data = s0[125];
                float v1504_data = ir0[7];
                ir0[7] = (v1504_data + (v1501_data * v1502_data));
              }
              if (v8_lead < 12) {
                float v1517_data = glb_m1[((v8_lead + 4) + 480)];
                float v1518_data = s0[15];
                float v1520_data = ir0[0];
                ir0[0] = (v1520_data + (v1517_data * v1518_data));
                float v1529_data = glb_m1[((v8_lead + 4) + 480)];
                float v1530_data = s0[31];
                float v1532_data = ir0[1];
                ir0[1] = (v1532_data + (v1529_data * v1530_data));
                float v1541_data = glb_m1[((v8_lead + 4) + 480)];
                float v1542_data = s0[46];
                float v1544_data = ir0[2];
                ir0[2] = (v1544_data + (v1541_data * v1542_data));
                float v1553_data = glb_m1[((v8_lead + 4) + 480)];
                float v1554_data = s0[62];
                float v1556_data = ir0[3];
                ir0[3] = (v1556_data + (v1553_data * v1554_data));
                float v1565_data = glb_m1[((v8_lead + 4) + 480)];
                float v1566_data = s0[77];
                float v1568_data = ir0[4];
                ir0[4] = (v1568_data + (v1565_data * v1566_data));
                float v1577_data = glb_m1[((v8_lead + 4) + 480)];
                float v1578_data = s0[93];
                float v1580_data = ir0[5];
                ir0[5] = (v1580_data + (v1577_data * v1578_data));
                float v1589_data = glb_m1[((v8_lead + 4) + 480)];
                float v1590_data = s0[108];
                float v1592_data = ir0[6];
                ir0[6] = (v1592_data + (v1589_data * v1590_data));
                float v1601_data = glb_m1[((v8_lead + 4) + 480)];
                float v1602_data = s0[124];
                float v1604_data = ir0[7];
                ir0[7] = (v1604_data + (v1601_data * v1602_data));
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1610_n1 = 0; v1610_n1 < 8; ++v1610_n1) {
                  float v1612_data = ir0[v1610_n1];
                  r0[v1610_n1] = v1612_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1618_i1 = 0; v1618_i1 < 8; ++v1618_i1) {
                  float v1620_data = r0[v1618_i1];
                  glb_m0[(v8_lead + (v1618_i1 * 12))] = v1620_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

