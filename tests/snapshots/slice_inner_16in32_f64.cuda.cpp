// === base name ===
kernel_5271f4f0047afd4a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5271f4f0047afd4a = {{16, 8, 1}, 16, 16, 1, 8, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5271f4f0047afd4a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5271f4f0047afd4a(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5271f4f0047afd4a(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5271f4f0047afd4a, block.x * block.y * block.z, 1152 * sizeof(double));
        CHECK_ERR;
        if (blocksPerSM > 0) {
          gridsize = smCount * blocksPerSM;
        }
        else {
          gridsize = smCount;
        }
      }
      
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 1152 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_5271f4f0047afd4a(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5271f4f0047afd4a(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_5271f4f0047afd4a, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_5271f4f0047afd4a<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_5271f4f0047afd4a(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 8 per block = block 16x8x1, 9216 B shared, occupancy grid
    // operands:
    //   m0 16×8(16×8) {0..16}×{0..8} strided
    //   m1 32×32(32×32) {0..32}×{0..32} strided
    //   m2 16×8(16×8) {0..16}×{0..8} strided
    // operations:
    //   m0[i,j] = m1[i,k]@{8..24}×{8..24} × m2[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1152}],"shared_bytes":9216,"shared_elements":1152,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,8]],"name":"m0","ordered":false,"parts":1,"shape":[16,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,32]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[8,8],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      double * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[v8_batchId0 * 128 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[v8_batchId0 * 1024 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[v8_batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v27_off = (v22_lead + (v23_i0 * 16)) + 8;
            #pragma unroll
            for (int32_t v24_i1 = 8; v24_i1 < 24; ++v24_i1) {
              double v30_data = __ldcg(&glb_m1[(v27_off + (v24_i1 * 32))]);
              r0[(v23_i0 + (v24_i1 - 8))] = v30_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 8);
          }
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          double r1[8]{};
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 8)] [(8, 24)]
          double ir1[8]{};
          double v36_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v37_data = s0[0];
          double v39_data = ir1[0];
          ir1[0] = (v39_data + (v36_data * v37_data));
          double v42_data = s0[16];
          double v44_data = ir1[1];
          ir1[1] = (v44_data + (v36_data * v42_data));
          double v47_data = s0[32];
          double v49_data = ir1[2];
          ir1[2] = (v49_data + (v36_data * v47_data));
          double v52_data = s0[48];
          double v54_data = ir1[3];
          ir1[3] = (v54_data + (v36_data * v52_data));
          double v57_data = s0[64];
          double v59_data = ir1[4];
          ir1[4] = (v59_data + (v36_data * v57_data));
          double v62_data = s0[80];
          double v64_data = ir1[5];
          ir1[5] = (v64_data + (v36_data * v62_data));
          double v67_data = s0[96];
          double v69_data = ir1[6];
          ir1[6] = (v69_data + (v36_data * v67_data));
          double v72_data = s0[112];
          double v74_data = ir1[7];
          ir1[7] = (v74_data + (v36_data * v72_data));
          double v76_data = r0[1];
          double v77_data = s0[1];
          double v79_data = ir1[0];
          ir1[0] = (v79_data + (v76_data * v77_data));
          double v82_data = s0[17];
          double v84_data = ir1[1];
          ir1[1] = (v84_data + (v76_data * v82_data));
          double v87_data = s0[33];
          double v89_data = ir1[2];
          ir1[2] = (v89_data + (v76_data * v87_data));
          double v92_data = s0[49];
          double v94_data = ir1[3];
          ir1[3] = (v94_data + (v76_data * v92_data));
          double v97_data = s0[65];
          double v99_data = ir1[4];
          ir1[4] = (v99_data + (v76_data * v97_data));
          double v102_data = s0[81];
          double v104_data = ir1[5];
          ir1[5] = (v104_data + (v76_data * v102_data));
          double v107_data = s0[97];
          double v109_data = ir1[6];
          ir1[6] = (v109_data + (v76_data * v107_data));
          double v112_data = s0[113];
          double v114_data = ir1[7];
          ir1[7] = (v114_data + (v76_data * v112_data));
          double v116_data = r0[2];
          double v117_data = s0[2];
          double v119_data = ir1[0];
          ir1[0] = (v119_data + (v116_data * v117_data));
          double v122_data = s0[18];
          double v124_data = ir1[1];
          ir1[1] = (v124_data + (v116_data * v122_data));
          double v127_data = s0[34];
          double v129_data = ir1[2];
          ir1[2] = (v129_data + (v116_data * v127_data));
          double v132_data = s0[50];
          double v134_data = ir1[3];
          ir1[3] = (v134_data + (v116_data * v132_data));
          double v137_data = s0[66];
          double v139_data = ir1[4];
          ir1[4] = (v139_data + (v116_data * v137_data));
          double v142_data = s0[82];
          double v144_data = ir1[5];
          ir1[5] = (v144_data + (v116_data * v142_data));
          double v147_data = s0[98];
          double v149_data = ir1[6];
          ir1[6] = (v149_data + (v116_data * v147_data));
          double v152_data = s0[114];
          double v154_data = ir1[7];
          ir1[7] = (v154_data + (v116_data * v152_data));
          double v156_data = r0[3];
          double v157_data = s0[3];
          double v159_data = ir1[0];
          ir1[0] = (v159_data + (v156_data * v157_data));
          double v162_data = s0[19];
          double v164_data = ir1[1];
          ir1[1] = (v164_data + (v156_data * v162_data));
          double v167_data = s0[35];
          double v169_data = ir1[2];
          ir1[2] = (v169_data + (v156_data * v167_data));
          double v172_data = s0[51];
          double v174_data = ir1[3];
          ir1[3] = (v174_data + (v156_data * v172_data));
          double v177_data = s0[67];
          double v179_data = ir1[4];
          ir1[4] = (v179_data + (v156_data * v177_data));
          double v182_data = s0[83];
          double v184_data = ir1[5];
          ir1[5] = (v184_data + (v156_data * v182_data));
          double v187_data = s0[99];
          double v189_data = ir1[6];
          ir1[6] = (v189_data + (v156_data * v187_data));
          double v192_data = s0[115];
          double v194_data = ir1[7];
          ir1[7] = (v194_data + (v156_data * v192_data));
          double v196_data = r0[4];
          double v197_data = s0[4];
          double v199_data = ir1[0];
          ir1[0] = (v199_data + (v196_data * v197_data));
          double v202_data = s0[20];
          double v204_data = ir1[1];
          ir1[1] = (v204_data + (v196_data * v202_data));
          double v207_data = s0[36];
          double v209_data = ir1[2];
          ir1[2] = (v209_data + (v196_data * v207_data));
          double v212_data = s0[52];
          double v214_data = ir1[3];
          ir1[3] = (v214_data + (v196_data * v212_data));
          double v217_data = s0[68];
          double v219_data = ir1[4];
          ir1[4] = (v219_data + (v196_data * v217_data));
          double v222_data = s0[84];
          double v224_data = ir1[5];
          ir1[5] = (v224_data + (v196_data * v222_data));
          double v227_data = s0[100];
          double v229_data = ir1[6];
          ir1[6] = (v229_data + (v196_data * v227_data));
          double v232_data = s0[116];
          double v234_data = ir1[7];
          ir1[7] = (v234_data + (v196_data * v232_data));
          double v236_data = r0[5];
          double v237_data = s0[5];
          double v239_data = ir1[0];
          ir1[0] = (v239_data + (v236_data * v237_data));
          double v242_data = s0[21];
          double v244_data = ir1[1];
          ir1[1] = (v244_data + (v236_data * v242_data));
          double v247_data = s0[37];
          double v249_data = ir1[2];
          ir1[2] = (v249_data + (v236_data * v247_data));
          double v252_data = s0[53];
          double v254_data = ir1[3];
          ir1[3] = (v254_data + (v236_data * v252_data));
          double v257_data = s0[69];
          double v259_data = ir1[4];
          ir1[4] = (v259_data + (v236_data * v257_data));
          double v262_data = s0[85];
          double v264_data = ir1[5];
          ir1[5] = (v264_data + (v236_data * v262_data));
          double v267_data = s0[101];
          double v269_data = ir1[6];
          ir1[6] = (v269_data + (v236_data * v267_data));
          double v272_data = s0[117];
          double v274_data = ir1[7];
          ir1[7] = (v274_data + (v236_data * v272_data));
          double v276_data = r0[6];
          double v277_data = s0[6];
          double v279_data = ir1[0];
          ir1[0] = (v279_data + (v276_data * v277_data));
          double v282_data = s0[22];
          double v284_data = ir1[1];
          ir1[1] = (v284_data + (v276_data * v282_data));
          double v287_data = s0[38];
          double v289_data = ir1[2];
          ir1[2] = (v289_data + (v276_data * v287_data));
          double v292_data = s0[54];
          double v294_data = ir1[3];
          ir1[3] = (v294_data + (v276_data * v292_data));
          double v297_data = s0[70];
          double v299_data = ir1[4];
          ir1[4] = (v299_data + (v276_data * v297_data));
          double v302_data = s0[86];
          double v304_data = ir1[5];
          ir1[5] = (v304_data + (v276_data * v302_data));
          double v307_data = s0[102];
          double v309_data = ir1[6];
          ir1[6] = (v309_data + (v276_data * v307_data));
          double v312_data = s0[118];
          double v314_data = ir1[7];
          ir1[7] = (v314_data + (v276_data * v312_data));
          double v316_data = r0[7];
          double v317_data = s0[7];
          double v319_data = ir1[0];
          ir1[0] = (v319_data + (v316_data * v317_data));
          double v322_data = s0[23];
          double v324_data = ir1[1];
          ir1[1] = (v324_data + (v316_data * v322_data));
          double v327_data = s0[39];
          double v329_data = ir1[2];
          ir1[2] = (v329_data + (v316_data * v327_data));
          double v332_data = s0[55];
          double v334_data = ir1[3];
          ir1[3] = (v334_data + (v316_data * v332_data));
          double v337_data = s0[71];
          double v339_data = ir1[4];
          ir1[4] = (v339_data + (v316_data * v337_data));
          double v342_data = s0[87];
          double v344_data = ir1[5];
          ir1[5] = (v344_data + (v316_data * v342_data));
          double v347_data = s0[103];
          double v349_data = ir1[6];
          ir1[6] = (v349_data + (v316_data * v347_data));
          double v352_data = s0[119];
          double v354_data = ir1[7];
          ir1[7] = (v354_data + (v316_data * v352_data));
          double v356_data = r0[8];
          double v357_data = s0[8];
          double v359_data = ir1[0];
          ir1[0] = (v359_data + (v356_data * v357_data));
          double v362_data = s0[24];
          double v364_data = ir1[1];
          ir1[1] = (v364_data + (v356_data * v362_data));
          double v367_data = s0[40];
          double v369_data = ir1[2];
          ir1[2] = (v369_data + (v356_data * v367_data));
          double v372_data = s0[56];
          double v374_data = ir1[3];
          ir1[3] = (v374_data + (v356_data * v372_data));
          double v377_data = s0[72];
          double v379_data = ir1[4];
          ir1[4] = (v379_data + (v356_data * v377_data));
          double v382_data = s0[88];
          double v384_data = ir1[5];
          ir1[5] = (v384_data + (v356_data * v382_data));
          double v387_data = s0[104];
          double v389_data = ir1[6];
          ir1[6] = (v389_data + (v356_data * v387_data));
          double v392_data = s0[120];
          double v394_data = ir1[7];
          ir1[7] = (v394_data + (v356_data * v392_data));
          double v396_data = r0[9];
          double v397_data = s0[9];
          double v399_data = ir1[0];
          ir1[0] = (v399_data + (v396_data * v397_data));
          double v402_data = s0[25];
          double v404_data = ir1[1];
          ir1[1] = (v404_data + (v396_data * v402_data));
          double v407_data = s0[41];
          double v409_data = ir1[2];
          ir1[2] = (v409_data + (v396_data * v407_data));
          double v412_data = s0[57];
          double v414_data = ir1[3];
          ir1[3] = (v414_data + (v396_data * v412_data));
          double v417_data = s0[73];
          double v419_data = ir1[4];
          ir1[4] = (v419_data + (v396_data * v417_data));
          double v422_data = s0[89];
          double v424_data = ir1[5];
          ir1[5] = (v424_data + (v396_data * v422_data));
          double v427_data = s0[105];
          double v429_data = ir1[6];
          ir1[6] = (v429_data + (v396_data * v427_data));
          double v432_data = s0[121];
          double v434_data = ir1[7];
          ir1[7] = (v434_data + (v396_data * v432_data));
          double v436_data = r0[10];
          double v437_data = s0[10];
          double v439_data = ir1[0];
          ir1[0] = (v439_data + (v436_data * v437_data));
          double v442_data = s0[26];
          double v444_data = ir1[1];
          ir1[1] = (v444_data + (v436_data * v442_data));
          double v447_data = s0[42];
          double v449_data = ir1[2];
          ir1[2] = (v449_data + (v436_data * v447_data));
          double v452_data = s0[58];
          double v454_data = ir1[3];
          ir1[3] = (v454_data + (v436_data * v452_data));
          double v457_data = s0[74];
          double v459_data = ir1[4];
          ir1[4] = (v459_data + (v436_data * v457_data));
          double v462_data = s0[90];
          double v464_data = ir1[5];
          ir1[5] = (v464_data + (v436_data * v462_data));
          double v467_data = s0[106];
          double v469_data = ir1[6];
          ir1[6] = (v469_data + (v436_data * v467_data));
          double v472_data = s0[122];
          double v474_data = ir1[7];
          ir1[7] = (v474_data + (v436_data * v472_data));
          double v476_data = r0[11];
          double v477_data = s0[11];
          double v479_data = ir1[0];
          ir1[0] = (v479_data + (v476_data * v477_data));
          double v482_data = s0[27];
          double v484_data = ir1[1];
          ir1[1] = (v484_data + (v476_data * v482_data));
          double v487_data = s0[43];
          double v489_data = ir1[2];
          ir1[2] = (v489_data + (v476_data * v487_data));
          double v492_data = s0[59];
          double v494_data = ir1[3];
          ir1[3] = (v494_data + (v476_data * v492_data));
          double v497_data = s0[75];
          double v499_data = ir1[4];
          ir1[4] = (v499_data + (v476_data * v497_data));
          double v502_data = s0[91];
          double v504_data = ir1[5];
          ir1[5] = (v504_data + (v476_data * v502_data));
          double v507_data = s0[107];
          double v509_data = ir1[6];
          ir1[6] = (v509_data + (v476_data * v507_data));
          double v512_data = s0[123];
          double v514_data = ir1[7];
          ir1[7] = (v514_data + (v476_data * v512_data));
          double v516_data = r0[12];
          double v517_data = s0[12];
          double v519_data = ir1[0];
          ir1[0] = (v519_data + (v516_data * v517_data));
          double v522_data = s0[28];
          double v524_data = ir1[1];
          ir1[1] = (v524_data + (v516_data * v522_data));
          double v527_data = s0[44];
          double v529_data = ir1[2];
          ir1[2] = (v529_data + (v516_data * v527_data));
          double v532_data = s0[60];
          double v534_data = ir1[3];
          ir1[3] = (v534_data + (v516_data * v532_data));
          double v537_data = s0[76];
          double v539_data = ir1[4];
          ir1[4] = (v539_data + (v516_data * v537_data));
          double v542_data = s0[92];
          double v544_data = ir1[5];
          ir1[5] = (v544_data + (v516_data * v542_data));
          double v547_data = s0[108];
          double v549_data = ir1[6];
          ir1[6] = (v549_data + (v516_data * v547_data));
          double v552_data = s0[124];
          double v554_data = ir1[7];
          ir1[7] = (v554_data + (v516_data * v552_data));
          double v556_data = r0[13];
          double v557_data = s0[13];
          double v559_data = ir1[0];
          ir1[0] = (v559_data + (v556_data * v557_data));
          double v562_data = s0[29];
          double v564_data = ir1[1];
          ir1[1] = (v564_data + (v556_data * v562_data));
          double v567_data = s0[45];
          double v569_data = ir1[2];
          ir1[2] = (v569_data + (v556_data * v567_data));
          double v572_data = s0[61];
          double v574_data = ir1[3];
          ir1[3] = (v574_data + (v556_data * v572_data));
          double v577_data = s0[77];
          double v579_data = ir1[4];
          ir1[4] = (v579_data + (v556_data * v577_data));
          double v582_data = s0[93];
          double v584_data = ir1[5];
          ir1[5] = (v584_data + (v556_data * v582_data));
          double v587_data = s0[109];
          double v589_data = ir1[6];
          ir1[6] = (v589_data + (v556_data * v587_data));
          double v592_data = s0[125];
          double v594_data = ir1[7];
          ir1[7] = (v594_data + (v556_data * v592_data));
          double v596_data = r0[14];
          double v597_data = s0[14];
          double v599_data = ir1[0];
          ir1[0] = (v599_data + (v596_data * v597_data));
          double v602_data = s0[30];
          double v604_data = ir1[1];
          ir1[1] = (v604_data + (v596_data * v602_data));
          double v607_data = s0[46];
          double v609_data = ir1[2];
          ir1[2] = (v609_data + (v596_data * v607_data));
          double v612_data = s0[62];
          double v614_data = ir1[3];
          ir1[3] = (v614_data + (v596_data * v612_data));
          double v617_data = s0[78];
          double v619_data = ir1[4];
          ir1[4] = (v619_data + (v596_data * v617_data));
          double v622_data = s0[94];
          double v624_data = ir1[5];
          ir1[5] = (v624_data + (v596_data * v622_data));
          double v627_data = s0[110];
          double v629_data = ir1[6];
          ir1[6] = (v629_data + (v596_data * v627_data));
          double v632_data = s0[126];
          double v634_data = ir1[7];
          ir1[7] = (v634_data + (v596_data * v632_data));
          double v636_data = r0[15];
          double v637_data = s0[15];
          double v639_data = ir1[0];
          ir1[0] = (v639_data + (v636_data * v637_data));
          double v642_data = s0[31];
          double v644_data = ir1[1];
          ir1[1] = (v644_data + (v636_data * v642_data));
          double v647_data = s0[47];
          double v649_data = ir1[2];
          ir1[2] = (v649_data + (v636_data * v647_data));
          double v652_data = s0[63];
          double v654_data = ir1[3];
          ir1[3] = (v654_data + (v636_data * v652_data));
          double v657_data = s0[79];
          double v659_data = ir1[4];
          ir1[4] = (v659_data + (v636_data * v657_data));
          double v662_data = s0[95];
          double v664_data = ir1[5];
          ir1[5] = (v664_data + (v636_data * v662_data));
          double v667_data = s0[111];
          double v669_data = ir1[6];
          ir1[6] = (v669_data + (v636_data * v667_data));
          double v672_data = s0[127];
          double v674_data = ir1[7];
          ir1[7] = (v674_data + (v636_data * v672_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v676_n0 = 0; v676_n0 < 1; ++v676_n0) {
            #pragma unroll
            for (int32_t v677_n1 = 0; v677_n1 < 8; ++v677_n1) {
              int32_t v678_a = v676_n0 + v677_n1;
              double v679_data = ir1[v678_a];
              r1[v678_a] = v679_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v680_i0 = 0; v680_i0 < 1; ++v680_i0) {
            int32_t v685_lead = v22_lead + (v680_i0 * 16);
            #pragma unroll
            for (int32_t v681_i1 = 0; v681_i1 < 8; ++v681_i1) {
              double v683_data = r1[(v680_i0 + v681_i1)];
              glb_m0[(v685_lead + (v681_i1 * 16))] = v683_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

