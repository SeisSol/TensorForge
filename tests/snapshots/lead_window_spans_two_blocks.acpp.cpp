// === base name ===
kernel_fe49398c33611f70

// === header ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
inline constexpr tensorforge::LaunchInfo launch_info_kernel_fe49398c33611f70 = {{32, 1, 1}, 32, 64, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_fe49398c33611f70(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_fe49398c33611f70(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
tensorforge::LaunchConfig launch_config_kernel_fe49398c33611f70(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (32, 1, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 1 - 1) / 1;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 1;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_fe49398c33611f70(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_fe49398c33611f70(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_fe49398c33611f70(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_fe49398c33611f70(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (64 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 64×13(64×13) {0..64}×{0..13} pointer_based
        //   m1 6(6) {0..6} none
        //   m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based
        // operations:
        //   t0[i,j,l] = m0[i,j] × m1[l]
        //   m2[i,j,l]@{20..35}×{12..13}×{0..6} += t0[i,j,l]@{20..35}×{12..13}×{0..6}
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"A","bbox":[[0,0],[64,13]],"name":"m0","ordered":false,"parts":1,"shape":[64,13],"variant":false},{"addressing":"none","alias":"v","bbox":[[0],[6]],"name":"m1","ordered":false,"parts":1,"shape":[6],"variant":false},{"addressing":"pointer_based","alias":"D","bbox":[[0,0,0],[64,13,6]],"name":"m2","ordered":false,"parts":1,"shape":[64,13,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0,0],[64,13,6]],"is_tmp":true,"name":"t0","offset":[0,0,0],"shape":[64,13,6]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[64,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,13]},{"addressing":"none","bbox":[[0],[6]],"is_tmp":false,"name":"m1","offset":[0],"shape":[6]}],"permute":[[0,1],[0]],"target":[[0,1],[2]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0,0],[15,1,6]],"is_tmp":false,"name":"m2","offset":[20,12,0],"shape":[64,13,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0,0],[15,1,6]],"is_tmp":true,"name":"t0","offset":[20,12,0],"shape":[64,13,6]}],"permute":[[0,1,2]],"target":[[0,1,2]]}],"version":"0.0.1"}
        {
          const float *const __restrict__ glb_m1 = &m1[0];
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v8_batchId0][0 + m0_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v8_batchId0][0 + m2_extraOffset];
              float r0[26]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v21_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v22_i0 = 0; v22_i0 < 2; ++v22_i0) {
                int32_t v25_lead = v21_lead + (v22_i0 * 32);
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 13; ++v23_i1) {
                  float v28_data = glb_m0[(v25_lead + (v23_i1 * 64))];
                  r0[(v22_i0 + (v23_i1 * 2))] = v28_data;
                }
              }
              float r2[12]{};
              // r2 = load{g>r}(glb_m2);
              bool v813_g = v21_lead >= 20;
              if (v813_g) {
                #pragma unroll
                for (int32_t v814_i1 = 0; v814_i1 < 1; ++v814_i1) {
                  int32_t v821_a = v21_lead + ((v814_i1 + 12) * 64);
                  int32_t v824_a = v814_i1 * 2;
                  #pragma unroll
                  for (int32_t v815_i2 = 0; v815_i2 < 6; ++v815_i2) {
                    float v823_data = glb_m2[(v821_a + (v815_i2 * 832))];
                    r2[(v824_a + (v815_i2 * 2))] = v823_data;
                  }
                }
              }
              bool v828_g = v21_lead < 3;
              if (v828_g) {
                int32_t v832_lead = v21_lead + 32_i32;
                #pragma unroll
                for (int32_t v829_i1 = 0; v829_i1 < 1; ++v829_i1) {
                  int32_t v836_a = v832_lead + ((v829_i1 + 12) * 64);
                  int32_t v841_a = 1 + (v829_i1 * 2);
                  #pragma unroll
                  for (int32_t v830_i2 = 0; v830_i2 < 6; ++v830_i2) {
                    float v838_data = glb_m2[(v836_a + (v830_i2 * 832))];
                    r2[(v841_a + (v830_i2 * 2))] = v838_data;
                  }
                }
              }
              float r1[156]{};
              // r1 = +(r0 * glb_m1) + None
              // [(0, 64), (0, 13), (0, 6)] []
              float v32_data = r0[0];
              float v33_data = glb_m1[0];
              float v35_data = r1[0];
              r1[0] = (v35_data + (v32_data * v33_data));
              float v38_data = glb_m1[1];
              float v40_data = r1[26];
              r1[26] = (v40_data + (v32_data * v38_data));
              float v43_data = glb_m1[2];
              float v45_data = r1[52];
              r1[52] = (v45_data + (v32_data * v43_data));
              float v48_data = glb_m1[3];
              float v50_data = r1[78];
              r1[78] = (v50_data + (v32_data * v48_data));
              float v53_data = glb_m1[4];
              float v55_data = r1[104];
              r1[104] = (v55_data + (v32_data * v53_data));
              float v58_data = glb_m1[5];
              float v60_data = r1[130];
              r1[130] = (v60_data + (v32_data * v58_data));
              float v62_data = r0[2];
              float v65_data = r1[2];
              r1[2] = (v65_data + (v62_data * v33_data));
              float v70_data = r1[28];
              r1[28] = (v70_data + (v62_data * v38_data));
              float v75_data = r1[54];
              r1[54] = (v75_data + (v62_data * v43_data));
              float v80_data = r1[80];
              r1[80] = (v80_data + (v62_data * v48_data));
              float v85_data = r1[106];
              r1[106] = (v85_data + (v62_data * v53_data));
              float v90_data = r1[132];
              r1[132] = (v90_data + (v62_data * v58_data));
              float v92_data = r0[4];
              float v95_data = r1[4];
              r1[4] = (v95_data + (v92_data * v33_data));
              float v100_data = r1[30];
              r1[30] = (v100_data + (v92_data * v38_data));
              float v105_data = r1[56];
              r1[56] = (v105_data + (v92_data * v43_data));
              float v110_data = r1[82];
              r1[82] = (v110_data + (v92_data * v48_data));
              float v115_data = r1[108];
              r1[108] = (v115_data + (v92_data * v53_data));
              float v120_data = r1[134];
              r1[134] = (v120_data + (v92_data * v58_data));
              float v122_data = r0[6];
              float v125_data = r1[6];
              r1[6] = (v125_data + (v122_data * v33_data));
              float v130_data = r1[32];
              r1[32] = (v130_data + (v122_data * v38_data));
              float v135_data = r1[58];
              r1[58] = (v135_data + (v122_data * v43_data));
              float v140_data = r1[84];
              r1[84] = (v140_data + (v122_data * v48_data));
              float v145_data = r1[110];
              r1[110] = (v145_data + (v122_data * v53_data));
              float v150_data = r1[136];
              r1[136] = (v150_data + (v122_data * v58_data));
              float v152_data = r0[8];
              float v155_data = r1[8];
              r1[8] = (v155_data + (v152_data * v33_data));
              float v160_data = r1[34];
              r1[34] = (v160_data + (v152_data * v38_data));
              float v165_data = r1[60];
              r1[60] = (v165_data + (v152_data * v43_data));
              float v170_data = r1[86];
              r1[86] = (v170_data + (v152_data * v48_data));
              float v175_data = r1[112];
              r1[112] = (v175_data + (v152_data * v53_data));
              float v180_data = r1[138];
              r1[138] = (v180_data + (v152_data * v58_data));
              float v182_data = r0[10];
              float v185_data = r1[10];
              r1[10] = (v185_data + (v182_data * v33_data));
              float v190_data = r1[36];
              r1[36] = (v190_data + (v182_data * v38_data));
              float v195_data = r1[62];
              r1[62] = (v195_data + (v182_data * v43_data));
              float v200_data = r1[88];
              r1[88] = (v200_data + (v182_data * v48_data));
              float v205_data = r1[114];
              r1[114] = (v205_data + (v182_data * v53_data));
              float v210_data = r1[140];
              r1[140] = (v210_data + (v182_data * v58_data));
              float v212_data = r0[12];
              float v215_data = r1[12];
              r1[12] = (v215_data + (v212_data * v33_data));
              float v220_data = r1[38];
              r1[38] = (v220_data + (v212_data * v38_data));
              float v225_data = r1[64];
              r1[64] = (v225_data + (v212_data * v43_data));
              float v230_data = r1[90];
              r1[90] = (v230_data + (v212_data * v48_data));
              float v235_data = r1[116];
              r1[116] = (v235_data + (v212_data * v53_data));
              float v240_data = r1[142];
              r1[142] = (v240_data + (v212_data * v58_data));
              float v242_data = r0[14];
              float v245_data = r1[14];
              r1[14] = (v245_data + (v242_data * v33_data));
              float v250_data = r1[40];
              r1[40] = (v250_data + (v242_data * v38_data));
              float v255_data = r1[66];
              r1[66] = (v255_data + (v242_data * v43_data));
              float v260_data = r1[92];
              r1[92] = (v260_data + (v242_data * v48_data));
              float v265_data = r1[118];
              r1[118] = (v265_data + (v242_data * v53_data));
              float v270_data = r1[144];
              r1[144] = (v270_data + (v242_data * v58_data));
              float v272_data = r0[16];
              float v275_data = r1[16];
              r1[16] = (v275_data + (v272_data * v33_data));
              float v280_data = r1[42];
              r1[42] = (v280_data + (v272_data * v38_data));
              float v285_data = r1[68];
              r1[68] = (v285_data + (v272_data * v43_data));
              float v290_data = r1[94];
              r1[94] = (v290_data + (v272_data * v48_data));
              float v295_data = r1[120];
              r1[120] = (v295_data + (v272_data * v53_data));
              float v300_data = r1[146];
              r1[146] = (v300_data + (v272_data * v58_data));
              float v302_data = r0[18];
              float v305_data = r1[18];
              r1[18] = (v305_data + (v302_data * v33_data));
              float v310_data = r1[44];
              r1[44] = (v310_data + (v302_data * v38_data));
              float v315_data = r1[70];
              r1[70] = (v315_data + (v302_data * v43_data));
              float v320_data = r1[96];
              r1[96] = (v320_data + (v302_data * v48_data));
              float v325_data = r1[122];
              r1[122] = (v325_data + (v302_data * v53_data));
              float v330_data = r1[148];
              r1[148] = (v330_data + (v302_data * v58_data));
              float v332_data = r0[20];
              float v335_data = r1[20];
              r1[20] = (v335_data + (v332_data * v33_data));
              float v340_data = r1[46];
              r1[46] = (v340_data + (v332_data * v38_data));
              float v345_data = r1[72];
              r1[72] = (v345_data + (v332_data * v43_data));
              float v350_data = r1[98];
              r1[98] = (v350_data + (v332_data * v48_data));
              float v355_data = r1[124];
              r1[124] = (v355_data + (v332_data * v53_data));
              float v360_data = r1[150];
              r1[150] = (v360_data + (v332_data * v58_data));
              float v362_data = r0[22];
              float v365_data = r1[22];
              r1[22] = (v365_data + (v362_data * v33_data));
              float v370_data = r1[48];
              r1[48] = (v370_data + (v362_data * v38_data));
              float v375_data = r1[74];
              r1[74] = (v375_data + (v362_data * v43_data));
              float v380_data = r1[100];
              r1[100] = (v380_data + (v362_data * v48_data));
              float v385_data = r1[126];
              r1[126] = (v385_data + (v362_data * v53_data));
              float v390_data = r1[152];
              r1[152] = (v390_data + (v362_data * v58_data));
              float v392_data = r0[24];
              float v395_data = r1[24];
              r1[24] = (v395_data + (v392_data * v33_data));
              float v400_data = r1[50];
              r1[50] = (v400_data + (v392_data * v38_data));
              float v405_data = r1[76];
              r1[76] = (v405_data + (v392_data * v43_data));
              float v410_data = r1[102];
              r1[102] = (v410_data + (v392_data * v48_data));
              float v415_data = r1[128];
              r1[128] = (v415_data + (v392_data * v53_data));
              float v420_data = r1[154];
              r1[154] = (v420_data + (v392_data * v58_data));
              float v422_data = r0[1];
              float v425_data = r1[1];
              r1[1] = (v425_data + (v422_data * v33_data));
              float v430_data = r1[27];
              r1[27] = (v430_data + (v422_data * v38_data));
              float v435_data = r1[53];
              r1[53] = (v435_data + (v422_data * v43_data));
              float v440_data = r1[79];
              r1[79] = (v440_data + (v422_data * v48_data));
              float v445_data = r1[105];
              r1[105] = (v445_data + (v422_data * v53_data));
              float v450_data = r1[131];
              r1[131] = (v450_data + (v422_data * v58_data));
              float v452_data = r0[3];
              float v455_data = r1[3];
              r1[3] = (v455_data + (v452_data * v33_data));
              float v460_data = r1[29];
              r1[29] = (v460_data + (v452_data * v38_data));
              float v465_data = r1[55];
              r1[55] = (v465_data + (v452_data * v43_data));
              float v470_data = r1[81];
              r1[81] = (v470_data + (v452_data * v48_data));
              float v475_data = r1[107];
              r1[107] = (v475_data + (v452_data * v53_data));
              float v480_data = r1[133];
              r1[133] = (v480_data + (v452_data * v58_data));
              float v482_data = r0[5];
              float v485_data = r1[5];
              r1[5] = (v485_data + (v482_data * v33_data));
              float v490_data = r1[31];
              r1[31] = (v490_data + (v482_data * v38_data));
              float v495_data = r1[57];
              r1[57] = (v495_data + (v482_data * v43_data));
              float v500_data = r1[83];
              r1[83] = (v500_data + (v482_data * v48_data));
              float v505_data = r1[109];
              r1[109] = (v505_data + (v482_data * v53_data));
              float v510_data = r1[135];
              r1[135] = (v510_data + (v482_data * v58_data));
              float v512_data = r0[7];
              float v515_data = r1[7];
              r1[7] = (v515_data + (v512_data * v33_data));
              float v520_data = r1[33];
              r1[33] = (v520_data + (v512_data * v38_data));
              float v525_data = r1[59];
              r1[59] = (v525_data + (v512_data * v43_data));
              float v530_data = r1[85];
              r1[85] = (v530_data + (v512_data * v48_data));
              float v535_data = r1[111];
              r1[111] = (v535_data + (v512_data * v53_data));
              float v540_data = r1[137];
              r1[137] = (v540_data + (v512_data * v58_data));
              float v542_data = r0[9];
              float v545_data = r1[9];
              r1[9] = (v545_data + (v542_data * v33_data));
              float v550_data = r1[35];
              r1[35] = (v550_data + (v542_data * v38_data));
              float v555_data = r1[61];
              r1[61] = (v555_data + (v542_data * v43_data));
              float v560_data = r1[87];
              r1[87] = (v560_data + (v542_data * v48_data));
              float v565_data = r1[113];
              r1[113] = (v565_data + (v542_data * v53_data));
              float v570_data = r1[139];
              r1[139] = (v570_data + (v542_data * v58_data));
              float v572_data = r0[11];
              float v575_data = r1[11];
              r1[11] = (v575_data + (v572_data * v33_data));
              float v580_data = r1[37];
              r1[37] = (v580_data + (v572_data * v38_data));
              float v585_data = r1[63];
              r1[63] = (v585_data + (v572_data * v43_data));
              float v590_data = r1[89];
              r1[89] = (v590_data + (v572_data * v48_data));
              float v595_data = r1[115];
              r1[115] = (v595_data + (v572_data * v53_data));
              float v600_data = r1[141];
              r1[141] = (v600_data + (v572_data * v58_data));
              float v602_data = r0[13];
              float v605_data = r1[13];
              r1[13] = (v605_data + (v602_data * v33_data));
              float v610_data = r1[39];
              r1[39] = (v610_data + (v602_data * v38_data));
              float v615_data = r1[65];
              r1[65] = (v615_data + (v602_data * v43_data));
              float v620_data = r1[91];
              r1[91] = (v620_data + (v602_data * v48_data));
              float v625_data = r1[117];
              r1[117] = (v625_data + (v602_data * v53_data));
              float v630_data = r1[143];
              r1[143] = (v630_data + (v602_data * v58_data));
              float v632_data = r0[15];
              float v635_data = r1[15];
              r1[15] = (v635_data + (v632_data * v33_data));
              float v640_data = r1[41];
              r1[41] = (v640_data + (v632_data * v38_data));
              float v645_data = r1[67];
              r1[67] = (v645_data + (v632_data * v43_data));
              float v650_data = r1[93];
              r1[93] = (v650_data + (v632_data * v48_data));
              float v655_data = r1[119];
              r1[119] = (v655_data + (v632_data * v53_data));
              float v660_data = r1[145];
              r1[145] = (v660_data + (v632_data * v58_data));
              float v662_data = r0[17];
              float v665_data = r1[17];
              r1[17] = (v665_data + (v662_data * v33_data));
              float v670_data = r1[43];
              r1[43] = (v670_data + (v662_data * v38_data));
              float v675_data = r1[69];
              r1[69] = (v675_data + (v662_data * v43_data));
              float v680_data = r1[95];
              r1[95] = (v680_data + (v662_data * v48_data));
              float v685_data = r1[121];
              r1[121] = (v685_data + (v662_data * v53_data));
              float v690_data = r1[147];
              r1[147] = (v690_data + (v662_data * v58_data));
              float v692_data = r0[19];
              float v695_data = r1[19];
              r1[19] = (v695_data + (v692_data * v33_data));
              float v700_data = r1[45];
              r1[45] = (v700_data + (v692_data * v38_data));
              float v705_data = r1[71];
              r1[71] = (v705_data + (v692_data * v43_data));
              float v710_data = r1[97];
              r1[97] = (v710_data + (v692_data * v48_data));
              float v715_data = r1[123];
              r1[123] = (v715_data + (v692_data * v53_data));
              float v720_data = r1[149];
              r1[149] = (v720_data + (v692_data * v58_data));
              float v722_data = r0[21];
              float v725_data = r1[21];
              r1[21] = (v725_data + (v722_data * v33_data));
              float v730_data = r1[47];
              r1[47] = (v730_data + (v722_data * v38_data));
              float v735_data = r1[73];
              r1[73] = (v735_data + (v722_data * v43_data));
              float v740_data = r1[99];
              r1[99] = (v740_data + (v722_data * v48_data));
              float v745_data = r1[125];
              r1[125] = (v745_data + (v722_data * v53_data));
              float v750_data = r1[151];
              r1[151] = (v750_data + (v722_data * v58_data));
              float v752_data = r0[23];
              float v755_data = r1[23];
              r1[23] = (v755_data + (v752_data * v33_data));
              float v760_data = r1[49];
              r1[49] = (v760_data + (v752_data * v38_data));
              float v765_data = r1[75];
              r1[75] = (v765_data + (v752_data * v43_data));
              float v770_data = r1[101];
              r1[101] = (v770_data + (v752_data * v48_data));
              float v775_data = r1[127];
              r1[127] = (v775_data + (v752_data * v53_data));
              float v780_data = r1[153];
              r1[153] = (v780_data + (v752_data * v58_data));
              float v782_data = r0[25];
              float v785_data = r1[25];
              r1[25] = (v785_data + (v782_data * v33_data));
              float v790_data = r1[51];
              r1[51] = (v790_data + (v782_data * v38_data));
              float v795_data = r1[77];
              r1[77] = (v795_data + (v782_data * v43_data));
              float v800_data = r1[103];
              r1[103] = (v800_data + (v782_data * v48_data));
              float v805_data = r1[129];
              r1[129] = (v805_data + (v782_data * v53_data));
              float v810_data = r1[155];
              r1[155] = (v810_data + (v782_data * v58_data));
              float r3[12]{};
              // ir3 = +(r1)
              // [(20, 35), (0, 1), (0, 6)] []
              float ir3[12]{};
              if (v813_g) {
                float v845_data = r1[24];
                float v846_data = ir3[0];
                ir3[0] = (v846_data + v845_data);
                float v848_data = r1[50];
                float v849_data = ir3[2];
                ir3[2] = (v849_data + v848_data);
                float v851_data = r1[76];
                float v852_data = ir3[4];
                ir3[4] = (v852_data + v851_data);
                float v854_data = r1[102];
                float v855_data = ir3[6];
                ir3[6] = (v855_data + v854_data);
                float v857_data = r1[128];
                float v858_data = ir3[8];
                ir3[8] = (v858_data + v857_data);
                float v860_data = r1[154];
                float v861_data = ir3[10];
                ir3[10] = (v861_data + v860_data);
              }
              if (v828_g) {
                float v863_data = r1[25];
                float v864_data = ir3[1];
                ir3[1] = (v864_data + v863_data);
                float v866_data = r1[51];
                float v867_data = ir3[3];
                ir3[3] = (v867_data + v866_data);
                float v869_data = r1[77];
                float v870_data = ir3[5];
                ir3[5] = (v870_data + v869_data);
                float v872_data = r1[103];
                float v873_data = ir3[7];
                ir3[7] = (v873_data + v872_data);
                float v875_data = r1[129];
                float v876_data = ir3[9];
                ir3[9] = (v876_data + v875_data);
                float v878_data = r1[155];
                float v879_data = ir3[11];
                ir3[11] = (v879_data + v878_data);
              }
              // r3 = ir3 + r2
              if (v813_g) {
                #pragma unroll
                for (int32_t v881_n1 = 0; v881_n1 < 1; ++v881_n1) {
                  int32_t v883_a = v881_n1 * 2;
                  #pragma unroll
                  for (int32_t v882_n2 = 0; v882_n2 < 6; ++v882_n2) {
                    int32_t v886_a = v883_a + (v882_n2 * 2);
                    float v887_data = ir3[v886_a];
                    float v888_data = r2[v886_a];
                    r3[v886_a] = (v888_data + v887_data);
                  }
                }
              }
              if (v828_g) {
                #pragma unroll
                for (int32_t v890_n1 = 0; v890_n1 < 1; ++v890_n1) {
                  int32_t v894_a = 1 + (v890_n1 * 2);
                  #pragma unroll
                  for (int32_t v891_n2 = 0; v891_n2 < 6; ++v891_n2) {
                    int32_t v895_a = v894_a + (v891_n2 * 2);
                    float v896_data = ir3[v895_a];
                    float v897_data = r2[v895_a];
                    r3[v895_a] = (v897_data + v896_data);
                  }
                }
              }
              // glb_m2 = store{r>g}(r3);
              if (v813_g) {
                #pragma unroll
                for (int32_t v899_i1 = 0; v899_i1 < 1; ++v899_i1) {
                  int32_t v901_a = v899_i1 * 2;
                  int32_t v911_a = v21_lead + ((v899_i1 + 12) * 64);
                  #pragma unroll
                  for (int32_t v900_i2 = 0; v900_i2 < 6; ++v900_i2) {
                    float v905_data = r3[(v901_a + (v900_i2 * 2))];
                    glb_m2[(v911_a + (v900_i2 * 832))] = v905_data;
                  }
                }
              }
              if (v828_g) {
                int32_t v921_lead = v21_lead + 32_i32;
                #pragma unroll
                for (int32_t v913_i1 = 0; v913_i1 < 1; ++v913_i1) {
                  int32_t v917_a = 1 + (v913_i1 * 2);
                  int32_t v925_a = v921_lead + ((v913_i1 + 12) * 64);
                  #pragma unroll
                  for (int32_t v914_i2 = 0; v914_i2 < 6; ++v914_i2) {
                    float v919_data = r3[(v917_a + (v914_i2 * 2))];
                    glb_m2[(v925_a + (v914_i2 * 832))] = v919_data;
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

