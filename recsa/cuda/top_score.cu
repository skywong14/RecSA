// RecSA fast top selection kernel A, hand-written for sm_90a.
#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda.h>

#ifndef NST
#define NST 2
#endif
#ifndef MINB
#define MINB 2
#endif
#ifndef ABL
#define ABL 0                   // timing ablations (wrong results): 1 epilogue skipped, 2 one wgmma per tile
#endif
#ifndef RSA
#define RSA 1                   // A / AR operands in registers (wgmma RS): no A tiles in shared memory, NST 3 fits 2 CTAs
#endif
#ifndef SACC
#define SACC 0                  // one accumulator: chain, wait, epilogue (overlap across CTAs only; for 3 CTAs per SM)
#endif
#ifndef RS
#define RS 1                    // head sum as a reduce-scatter (24 shuffles per tile instead of 64; same adds)
#endif
#define QB 16
#define GH 4
#define TN 64
#define ATILE 16384             // 64 rows x 128 fp16: two 8 KB halves (dims 0-63 | 64-127), 128B-swizzled
#define STAGEB 32768            // K tile | K-residual tile
#define LOG2E_F 1.4426950408889634f

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ uint32_t mbar_try(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred P1;\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n selp.u32 %0, 1, 0, P1;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) { while (!mbar_try(bar, parity)) {} }
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma3(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               :: "r"(dst), "l"((uint64_t)map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
__device__ __forceinline__ float ex2f(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float ex2f_ftz(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void wg_keep(float (&d)[32]) {
#pragma unroll
  for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}
// K-major operand, 128B swizzle: rows of 128 B in 1024 B atoms of 8 rows (SBO 1024), LBO unused (1)
__device__ __forceinline__ uint64_t mdesc(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
template <int SD>
__device__ __forceinline__ void wgmma64(float (&d)[32], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %34, 0;\n"
      " wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "n"(SD));
}
#define DHI ((uint32_t)(1024 >> 4) | (1u << 30))
__device__ __forceinline__ uint32_t dlo(uint32_t addr) { return ((addr & 0x3FFFF) >> 4) | (1u << 16); }
template <int SD>
__device__ __forceinline__ void wgmma64r(float (&d)[32], const uint32_t (&a)[4], uint32_t blo) {
  asm volatile(
      "{\n .reg .pred p;\n .reg .b64 db;\n setp.ne.b32 p, %38, 0;\n mov.b64 db, {%36, %37};\n"
      " wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "{%32, %33, %34, %35}, db, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(blo), "r"(DHI), "n"(SD));
}
// RS form: the same 24 wgmma with the A / AR fragments in registers; K / KR descriptors as base + k-step offset
__device__ __forceinline__ void chain_r(float (&d)[32], const uint32_t (&aF)[8][4], const uint32_t (&arF)[8][4],
                                        uint32_t sK, uint32_t sKR) {
  const uint32_t lk = dlo(sK), lr = dlo(sKR);
  wg_fence();
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    const uint32_t o = (kk >> 2) * 512 + (kk & 3) * 2;
    if (kk == 0) wgmma64r<0>(d, aF[kk], lk + o);
    else wgmma64r<1>(d, aF[kk], lk + o);
  }
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) wgmma64r<1>(d, aF[kk], lr + (kk >> 2) * 512 + (kk & 3) * 2);
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) wgmma64r<1>(d, arF[kk], lk + (kk >> 2) * 512 + (kk & 3) * 2);
  wg_commit();
}
// the 24 wgmma of one tile: A.K, A.KR, AR.K (k16 steps in order), one accumulator, the first with scale-d 0
__device__ __forceinline__ void chain(float (&d)[32], uint32_t sA, uint32_t sAR, uint32_t sK, uint32_t sKR) {
  wg_fence();
  if (ABL == 2) {
    wgmma64<0>(d, mdesc(sA), mdesc(sK));
    wg_commit();
    return;
  }
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    const uint32_t o = (kk >> 2) * 8192 + (kk & 3) * 32;
    if (kk == 0) wgmma64<0>(d, mdesc(sA + o), mdesc(sK + o));
    else wgmma64<1>(d, mdesc(sA + o), mdesc(sK + o));
  }
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    const uint32_t o = (kk >> 2) * 8192 + (kk & 3) * 32;
    wgmma64<1>(d, mdesc(sA + o), mdesc(sKR + o));
  }
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    const uint32_t o = (kk >> 2) * 8192 + (kk & 3) * 32;
    wgmma64<1>(d, mdesc(sAR + o), mdesc(sK + o));
  }
  wg_commit();
}
__device__ __forceinline__ uint32_t sw128(int r, int cc) {     // byte offset of (row, col < 64) in a swizzled half
  return r * 128 + ((((cc >> 3) ^ (r & 7))) << 4) + (cc & 7) * 2;
}

extern "C" __global__ void __launch_bounds__(128, MINB)
top_score_cuda(const __nv_bfloat16* __restrict__ q, const float* __restrict__ cosc, const float* __restrict__ sinc,
               float* __restrict__ raw, float* __restrict__ pmax,
               int t_base, int q0, int T, int QC, int N_layer, int NP, int NT, int layer_power, int H, int Hkv, int Bn,
               float scale, float neg_inf, const __grid_constant__ CUtensorMap tmk) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
  const uint32_t sb = smem_u32(smem);
  const uint32_t sA = sb, sAR = sb + ATILE, sS = sb + (RSA ? 0 : 2 * ATILE);
  const uint32_t bar0 = sS + NST * STAGEB;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int i_blk = blockIdx.x, pid = blockIdx.y;
  const int i_b = pid / Hkv, i_h = pid % Hkv;
  const int nmax = (t_base + min(q0 + i_blk * QB + QB - 1, T - 1)) / layer_power + 1;
  const int n_t = (nmax + TN - 1) / TN;
  const int rowk0 = i_b * N_layer, rowkr0 = Bn * N_layer + i_b * N_layer;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) mbar_init(bar0 + 8 * s, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    for (int s = 0; s < NST && s < n_t; ++s) {
      const uint32_t st = sS + s * STAGEB, bar = bar0 + 8 * s;
      mbar_expect_tx(bar, STAGEB);
      tma3(st, &tmk, 0, i_h, rowk0 + s * TN, bar);
      tma3(st + 8192, &tmk, 64, i_h, rowk0 + s * TN, bar);
      tma3(st + 16384, &tmk, 0, i_h, rowkr0 + s * TN, bar);
      tma3(st + 24576, &tmk, 64, i_h, rowkr0 + s * TN, bar);
    }
  }
#if RSA
  // ---- A / AR fragments in registers: rows 16 w + lane / 4 (+ 8), k columns 16 kk + 2 (lane % 4) + {0, 1} (+ 8) ----
  uint32_t aF[8][4], arF[8][4];
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    const int r = 16 * w + (lane >> 2) + 8 * x, j = r >> 2, g = r & 3;
    const int lq = i_blk * QB + j;
    const bool ok = (lq < QC) && (q0 + lq < T);
    const int t_loc = min(q0 + lq, T - 1);
    const int pos = (t_base + t_loc) / layer_power;
    const __nv_bfloat16* qp = q + (((long)i_b * T + t_loc) * H + (i_h * GH + g)) * 128;
    const float* cp = cosc + (long)pos * 64;
    const float* sp = sinc + (long)pos * 64;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int f = 16 * kk + 8 * hh + 2 * (lane & 3);
        const __nv_bfloat162 q2l = ok ? *reinterpret_cast<const __nv_bfloat162*>(qp + f) : __floats2bfloat162_rn(0.f, 0.f);
        const __nv_bfloat162 q2h = ok ? *reinterpret_cast<const __nv_bfloat162*>(qp + f + 64) : __floats2bfloat162_rn(0.f, 0.f);
        const float2 c2 = *reinterpret_cast<const float2*>(cp + f);
        const float2 s2 = *reinterpret_cast<const float2*>(sp + f);
        __half hl[2], hh2[2], rl[2], rh[2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const float ql = __bfloat162float(u ? q2l.y : q2l.x), qh = __bfloat162float(u ? q2h.y : q2h.x);
          const float c = u ? c2.y : c2.x, sn = u ? s2.y : s2.x;
          const float xl = __fmaf_rn(ql, c, -__fmul_rn(qh, sn));
          const float xh = __fmaf_rn(qh, c, __fmul_rn(ql, sn));
          hl[u] = __float2half_rn(__fmul_rn(scale, xl));
          hh2[u] = __float2half_rn(__fmul_rn(scale, xh));
          rl[u] = __float2half_rn(__fmaf_rn(scale, xl, -__half2float(hl[u])));
          rh[u] = __float2half_rn(__fmaf_rn(scale, xh, -__half2float(hh2[u])));
        }
        // fragment register: x (row half) + 2 hh (k half); k-step kk for the lo dims, kk + 4 for the hi dims
        const int ri = x + 2 * hh;
        aF[kk][ri] = (uint32_t)__half_as_ushort(hl[0]) | ((uint32_t)__half_as_ushort(hl[1]) << 16);
        aF[kk + 4][ri] = (uint32_t)__half_as_ushort(hh2[0]) | ((uint32_t)__half_as_ushort(hh2[1]) << 16);
        arF[kk][ri] = (uint32_t)__half_as_ushort(rl[0]) | ((uint32_t)__half_as_ushort(rl[1]) << 16);
        arF[kk + 4][ri] = (uint32_t)__half_as_ushort(rh[0]) | ((uint32_t)__half_as_ushort(rh[1]) << 16);
      }
  }
  __syncthreads();
#else
  // ---- A / AR: row r = 4 j + g (query j, head g), this thread's 32 frequencies [32 hf, 32 hf + 32) ----
  {
    const int r = tid >> 1, hf = tid & 1, j = r >> 2, g = r & 3;
    const int lq = i_blk * QB + j;
    const bool ok = (lq < QC) && (q0 + lq < T);
    const int t_loc = min(q0 + lq, T - 1);
    const int pos = (t_base + t_loc) / layer_power;
    const __nv_bfloat16* qp = q + (((long)i_b * T + t_loc) * H + (i_h * GH + g)) * 128;
    const float* cp = cosc + (long)pos * 64;
    const float* sp = sinc + (long)pos * 64;
#pragma unroll
    for (int ch = 0; ch < 4; ++ch) {
      __align__(16) __half alo[8], ahi[8], rlo[8], rhi[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int f = hf * 32 + ch * 8 + u;
        const float ql = ok ? __bfloat162float(qp[f]) : 0.f;
        const float qh = ok ? __bfloat162float(qp[f + 64]) : 0.f;
        const float c = cp[f], s = sp[f];
        const float xl = __fmaf_rn(ql, c, -__fmul_rn(qh, s));
        const float xh = __fmaf_rn(qh, c, __fmul_rn(ql, s));
        const __half hl = __float2half_rn(__fmul_rn(scale, xl));
        const __half hh = __float2half_rn(__fmul_rn(scale, xh));
        alo[u] = hl;
        ahi[u] = hh;
        rlo[u] = __float2half_rn(__fmaf_rn(scale, xl, -__half2float(hl)));
        rhi[u] = __float2half_rn(__fmaf_rn(scale, xh, -__half2float(hh)));
      }
      const int cc = hf * 32 + ch * 8;
      const uint32_t off = sw128(r, cc);
      *reinterpret_cast<uint4*>(smem + off) = *reinterpret_cast<uint4*>(alo);
      *reinterpret_cast<uint4*>(smem + 8192 + off) = *reinterpret_cast<uint4*>(ahi);
      *reinterpret_cast<uint4*>(smem + ATILE + off) = *reinterpret_cast<uint4*>(rlo);
      *reinterpret_cast<uint4*>(smem + ATILE + 8192 + off) = *reinterpret_cast<uint4*>(rhi);
    }
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
#endif

  // ---- this thread's accumulator rows: ra = 16 w + lane / 4 (query ja, head hd) and ra + 8 (query ja + 2) ----
  const int hd = (lane >> 2) & 3;
  const int ja = 4 * w + (lane >> 4), jb = ja + 2;
  int lim[2];          // valid columns of the row: node < ncand (0 for a masked row)
  bool qok[2];
  long rbase[2], mbase[2];
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    const int lq = i_blk * QB + (x ? jb : ja);
    qok[x] = (lq < QC) && (q0 + lq < T);
    const int t_loc = min(q0 + lq, T - 1);
    lim[x] = qok[x] ? (t_base + t_loc) / layer_power + 1 : 0;
    rbase[x] = ((long)pid * QC + lq) * NP;
    mbase[x] = ((long)pid * QC + lq) * NT;
  }
  const int cb = (lane & 3) * 2;
  int lmin = min(lim[0], lim[1]);
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) lmin = min(lmin, __shfl_xor_sync(0xffffffffu, lmin, off));
  // CTA-uniform: the smallest row limit (masked rows count as 0 only when they exist)
  __shared__ int s_lmin[4];
  if (lane == 0) s_lmin[w] = lmin;
  __syncthreads();
  lmin = min(min(s_lmin[0], s_lmin[1]), min(s_lmin[2], s_lmin[3]));

  // per-row output pointers: this lane stores blocks hd and hd + 4 (columns 8 hd + cb, + 32) of rows ja / jb
  float* rp[2];
  float* pp[2];
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    rp[x] = raw + rbase[x] + 8 * hd + cb;
    pp[x] = pmax + mbase[x];
  }
  // head sum (e_h + e_h^2) + (e_h^1 + e_h^3) as a reduce-scatter (this lane keeps blocks hd, hd + 4), then the stores
  auto finish = [&](const float (&e)[32], const float (&m)[2], int it) {
    const bool b1 = (hd >> 1) & 1, b0 = hd & 1;
    float p1[16];        // step 1 (xor 8): blocks with (i & 2) == (hd & 2)
#pragma unroll
    for (int ii = 0; ii < 4; ++ii) {
      const int i = (ii & 1) + ((ii >> 1) << 2);          // 0, 1, 4, 5: blocks with bit 1 clear
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float lo = e[4 * i + c], hi = e[4 * (i + 2) + c];
        const float keep = b1 ? hi : lo, send = b1 ? lo : hi;
        p1[4 * ii + c] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
      }
    }
    float p2[8];         // step 2 (xor 4): of those, blocks with (i & 1) == (hd & 1)
#pragma unroll
    for (int h4 = 0; h4 < 2; ++h4)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float lo = p1[4 * (2 * h4) + c], hi = p1[4 * (2 * h4 + 1) + c];
        const float keep = b0 ? hi : lo, send = b0 ? lo : hi;
        p2[4 * h4 + c] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
      }
    const int node0 = it * TN;
#pragma unroll
    for (int x = 0; x < 2; ++x)
      if (qok[x]) {
        *reinterpret_cast<float2*>(rp[x] + node0) = make_float2(p2[2 * x], p2[2 * x + 1]);
        *reinterpret_cast<float2*>(rp[x] + node0 + 32) = make_float2(p2[4 + 2 * x], p2[4 + 2 * x + 1]);
      }
    if ((lane & 15) == 0) {
#pragma unroll
      for (int x = 0; x < 2; ++x)
        if (qok[x]) pp[x][it] = m[x];
    }
  };
  auto rowmax = [&](const float (&v)[32], float (&m)[2]) {
#pragma unroll
    for (int x = 0; x < 2; ++x) {
      float t[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) t[i] = fmaxf(v[4 * i + 2 * x], v[4 * i + 2 * x + 1]);
#pragma unroll
      for (int i = 0; i < 4; ++i) t[i] = fmaxf(t[i], t[i + 4]);
      float r = fmaxf(fmaxf(t[0], t[2]), fmaxf(t[1], t[3]));
      r = fmaxf(r, __shfl_xor_sync(0xffffffffu, r, 1));
      r = fmaxf(r, __shfl_xor_sync(0xffffffffu, r, 2));
      r = fmaxf(r, __shfl_xor_sync(0xffffffffu, r, 4));
      r = fmaxf(r, __shfl_xor_sync(0xffffffffu, r, 8));
      m[x] = r;
    }
  };
  auto epilogue = [&](const float (&acc)[32], int it) {
    // reads the accumulator only (no non-wgmma definitions of wgmma registers: keeps the wgmma pipeline async)
    const int node0 = it * TN;
    if (ABL == 1) {
      if (acc[0] == 12345.f && qok[0]) raw[rbase[0] + node0] = acc[1];
      return;
    }
    float e[32], m[2];
    if (node0 + TN <= lmin) {           // CTA-uniform: every score of the tile is valid
      rowmax(acc, m);
      // ex2.approx (non-ftz) == ex2.approx.ftz when every argument is >= -126 (no denormal results): the argument is
      // monotone in the score, so the own minimum decides for the whole thread
      bool ok = true;
#pragma unroll
      for (int x = 0; x < 2; ++x) {
        float t[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) t[i] = fminf(acc[4 * i + 2 * x], acc[4 * i + 2 * x + 1]);
#pragma unroll
        for (int i = 0; i < 4; ++i) t[i] = fminf(t[i], t[i + 4]);
        const float mn = fminf(fminf(t[0], t[2]), fminf(t[1], t[3]));
        ok = ok && __fmul_rn(__fsub_rn(mn, m[x]), LOG2E_F) >= -126.f;
      }
      if (__all_sync(0xffffffffu, ok)) {
#pragma unroll
        for (int i = 0; i < 32; ++i) e[i] = ex2f_ftz(__fmul_rn(__fsub_rn(acc[i], m[(i >> 1) & 1]), LOG2E_F));
      } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) e[i] = ex2f(__fmul_rn(__fsub_rn(acc[i], m[(i >> 1) & 1]), LOG2E_F));
      }
    } else {
      float d[32];
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int col = node0 + 8 * i + cb + (c & 1);
          d[4 * i + c] = col < lim[c >> 1] ? acc[4 * i + c] : neg_inf;
        }
      rowmax(d, m);
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int col = node0 + 8 * i + cb + (c & 1);
          const float v = ex2f(__fmul_rn(__fsub_rn(d[4 * i + c], m[c >> 1]), LOG2E_F));
          e[4 * i + c] = col < lim[c >> 1] ? v : 0.f;
        }
    }
    finish(e, m, it);
  };
  auto refill = [&](int jt) {          // stage of tile jt (its chain complete in every warp) <- tile jt + NST
    if (jt + NST < n_t) {
      asm volatile("bar.sync 1, 128;" ::: "memory");
      if (tid == 0) {
        const int s = jt % NST, nt = jt + NST;
        const uint32_t st = sS + s * STAGEB, bar = bar0 + 8 * s;
        mbar_expect_tx(bar, STAGEB);
        tma3(st, &tmk, 0, i_h, rowk0 + nt * TN, bar);
        tma3(st + 8192, &tmk, 64, i_h, rowk0 + nt * TN, bar);
        tma3(st + 16384, &tmk, 0, i_h, rowkr0 + nt * TN, bar);
        tma3(st + 24576, &tmk, 64, i_h, rowkr0 + nt * TN, bar);
      }
    }
  };
  float acc0[32], acc1[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) acc0[i] = acc1[i] = 0.f;
#if SACC
  for (int jt = 0; jt < n_t; ++jt) {
    const int st = jt % NST;
    mbar_wait(bar0 + 8 * st, (jt / NST) & 1);
#if RSA
    chain_r(acc0, aF, arF, sS + st * STAGEB, sS + st * STAGEB + 16384);
#else
    chain(acc0, sA, sAR, sS + st * STAGEB, sS + st * STAGEB + 16384);
#endif
    wg_wait<0>();
    refill(jt);
    epilogue(acc0, jt);
  }
#else
  // FA3-style: the next tile's chain is issued, this tile's epilogue runs, then wait_group 0 (nothing in flight at
  // the loop back-edge, so ptxas keeps the wgmma pipeline asynchronous)
  if (n_t > 0) {
    mbar_wait(bar0, 0);
#if RSA
    chain_r(acc0, aF, arF, sS, sS + 16384);
#else
    chain(acc0, sA, sAR, sS, sS + 16384);
#endif
    wg_wait<0>();
  }
  for (int jt = 0; jt < n_t; jt += 2) {
    refill(jt);
    if (jt + 1 < n_t) {
      const int s = (jt + 1) % NST;
      mbar_wait(bar0 + 8 * s, ((jt + 1) / NST) & 1);
#if RSA
      chain_r(acc1, aF, arF, sS + s * STAGEB, sS + s * STAGEB + 16384);
#else
      chain(acc1, sA, sAR, sS + s * STAGEB, sS + s * STAGEB + 16384);
#endif
    }
    epilogue(acc0, jt);
    wg_wait<0>();
    if (jt + 1 >= n_t) break;
    refill(jt + 1);
    if (jt + 2 < n_t) {
      const int s = (jt + 2) % NST;
      mbar_wait(bar0 + 8 * s, ((jt + 2) / NST) & 1);
#if RSA
      chain_r(acc0, aF, arF, sS + s * STAGEB, sS + s * STAGEB + 16384);
#else
      chain(acc0, sA, sAR, sS + s * STAGEB, sS + s * STAGEB + 16384);
#endif
    }
    epilogue(acc1, jt + 1);
    wg_wait<0>();
  }
#endif
}
