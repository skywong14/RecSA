// RecSA prefill leaf union walk, hand-written for sm_90a.
#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda.h>

#ifndef NWARP
#define NWARP 8
#endif
#ifndef NSLOT
#define NSLOT 16
#endif
#ifndef EG
#define EG 4
#endif
#ifndef LMAX
#define LMAX 0
#endif
#ifndef USE_TMA
#define USE_TMA 1
#endif
#ifndef NO_WAIT
#define NO_WAIT 0
#endif
#ifndef MINB
#define MINB (8 / NWARP)
#endif
#ifndef C_SMEM
#define C_SMEM 0               // R(-CR) coefficients read from shared memory (frees 32 registers)
#endif
#ifndef SPLITK
#define SPLITK 0               // lo / hi halves of the QK in two MMA chains, then added
#endif
#ifndef ROT_EARLY
#define ROT_EARLY 0             // rotate the member rows right after the QK MMAs are issued (overlaps their latency)
#endif
#ifndef ROT_SEL
#define ROT_SEL 0               // branch-free rotation: rotate both rows, select by membership
#endif
#ifndef ABL
#define ABL 0                  // diagnostic ablations (wrong math): 1 no rotation, 2 no PV, 3 no QK MMA, 4 no exp, 5 no member steps
#endif
#ifndef LOOKAHEAD
#define LOOKAHEAD 0
#endif
#ifndef GL
#define GL 0                   // group-vectorized member masks / parent ids, member entries only (needs EG 4, UMAX % 4 0)
#endif
#if GL && (EG != 4 || LOOKAHEAD || LMAX)
#error "GL needs EG 4, LOOKAHEAD 0, LMAX 0"
#endif
#ifndef SKIP_RESCALE
#define SKIP_RESCALE 1
#endif
#ifndef FAST_MASK
#define FAST_MASK 1
#endif
#define NGRP (NSLOT / EG)
#define BQB (4 * NWARP)
#define CRN 16
#if USE_TMA
#define SLOTB 8192                  // K lo | K hi | V lo | V hi, 16 rows x 128 B each, TMA 128B swizzle
#else
#define ROWB 272                    // smem row pitch (256 B + 16 B: conflict-free ldmatrix)
#define TILEB (CRN * ROWB)
#define SLOTB (2 * TILEB)
#endif
#define LOG2E 1.4426950408889634f

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
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float ex2f(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
#ifndef EXFTZ
#define EXFTZ 1                 // softmax exps as ex2.approx.ftz (results below 2^-126 flushed)
#endif
__device__ __forceinline__ float ex2s(float x) {
  float y;
#if EXFTZ
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
#else
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
#endif
  return y;
}
#if !USE_TMA
// one warp fills the EG slots of a ring group with entries e0 .. e0 + EG - 1 (16 B cp.async per lane), then arrives
__device__ __forceinline__ void fill_group(uint32_t sb, int g, int e0, int n_iter, const int* lp, uint32_t bar,
                                           const __half* krot, const __half* vl, long tokb, int N_layer, int Hkv,
                                           int i_h, int lane) {
#pragma unroll
  for (int j = 0; j < EG; ++j) {
    const int e = e0 + j;
    if (e < n_iter) {
      const int p = lp[e];
      const uint32_t ss = sb + (g * EG + j) * SLOTB;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int c = lane + 32 * i, tile = c >> 8, row = (c & 255) >> 4, c16 = c & 15;
        const long tok = tokb + min(p * CRN + row, N_layer - 1);
        cp16(ss + tile * TILEB + row * ROWB + c16 * 16, (tile ? vl : krot) + (tok * Hkv + i_h) * 128 + c16 * 8);
      }
    }
  }
  cp_arrive(bar);
}
#endif

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma3(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               :: "r"(dst), "l"((uint64_t)map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
// one thread fills the ring group g with entries e0 .. e0 + EG - 1: 4 swizzled TMA boxes (64 dims x 16 tokens) each
__device__ __forceinline__ void fill_group_tma(uint32_t sb, int g, int e0, int n_iter, const int* lp, uint32_t bar,
                                               const CUtensorMap* mk, const CUtensorMap* mv, int tok0, int i_h) {
  const int ne = min(EG, n_iter - e0);
  mbar_expect_tx(bar, ne * SLOTB);
  for (int j = 0; j < ne; ++j) {
    const int tok = tok0 + lp[e0 + j] * CRN;
    const uint32_t ss = sb + (g * EG + j) * SLOTB;
    tma3(ss, mk, 0, i_h, tok, bar);
    tma3(ss + 2048, mk, 64, i_h, tok, bar);
    tma3(ss + 4096, mv, 0, i_h, tok, bar);
    tma3(ss + 6144, mv, 64, i_h, tok, bar);
  }
}

__device__ __forceinline__ void touch_row(float (&a)[8][4]) {
#pragma unroll
  for (int kb = 0; kb < 8; ++kb)
#pragma unroll
    for (int i = 0; i < 4; ++i) asm volatile("" : "+f"(a[kb][i]));
}
// rotate one row's 16 (lo, hi) pairs by R(-CR) with coefficients from shared memory (C_SMEM)
__device__ __forceinline__ void rot_row_sm(float (&a)[8][4], const float* csm, int cc) {
#pragma unroll
  for (int kb = 0; kb < 4; ++kb)
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      const int f = kb * 16 + cc + h2 * 8;
#if ABL == 7
      const float cs[2] = {0.9f + f * 1e-3f, 0.8f}, sn[2] = {0.1f, 0.2f};     // timing only: no coefficient loads
#else
      const float2 c2 = *reinterpret_cast<const float2*>(csm + f);
      const float2 s2 = *reinterpret_cast<const float2*>(csm + 64 + f);
      const float cs[2] = {c2.x, c2.y}, sn[2] = {s2.x, s2.y};
#endif
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int i = h2 * 2 + u;
        const float lo = a[kb][i], hi = a[kb + 4][i];
        a[kb][i] = lo * cs[u] + hi * sn[u];
        a[kb + 4][i] = hi * cs[u] - lo * sn[u];
      }
    }
}
// rotate one row's 16 (lo, hi) pairs by R(-CR): lo' = lo c + hi s, hi' = hi c - lo s
__device__ __forceinline__ void rot_row(float (&a)[8][4], const float (&c)[4][4], const float (&s)[4][4]) {
#pragma unroll
  for (int kb = 0; kb < 4; ++kb)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float lo = a[kb][i], hi = a[kb + 4][i];
      a[kb][i] = lo * c[kb][i] + hi * s[kb][i];
      a[kb + 4][i] = hi * c[kb][i] - lo * s[kb][i];
    }
}

#if C_SMEM
#define ROT(x) do { if (ABL == 6) touch_row(x); else if (ABL != 1) rot_row_sm(x, csm, cc); } while (0)
#else
#define ROT(x) do { if (ABL == 6) touch_row(x); else if (ABL != 1) rot_row(x, c16, s16); } while (0)
#endif

extern "C" __global__ void __launch_bounds__(32 * NWARP, MINB)
leaf_union_cuda(const __nv_bfloat16* __restrict__ q, const __half* __restrict__ krot, const __half* __restrict__ vl,
                const int* __restrict__ ulist, const int* __restrict__ umask, const int* __restrict__ ucount,
                const int* __restrict__ posq, const float* __restrict__ cosc, const float* __restrict__ sinc,
                float* __restrict__ out_max, float* __restrict__ out_sum, float* __restrict__ out_o,
                int t_base, int T, int N_layer, int UMAX, int H, int Hkv, float scale, float neg_inf,
                const __grid_constant__ CUtensorMap tmk, const __grid_constant__ CUtensorMap tmv,
                const float* __restrict__ gmax, const float* __restrict__ gsum, const float* __restrict__ gout,
                __nv_bfloat16* __restrict__ fout, int FUSE) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
  int* cnt = (int*)(smem + NSLOT * SLOTB + NGRP * 8);
  int* lsm = cnt + NGRP;
  float* csm = (float*)(((uintptr_t)(lsm + 2 * LMAX) + 15) & ~(uintptr_t)15);   // C_SMEM: cos / sin of R(CR) [2][64]
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int i_blk = blockIdx.x, pid = blockIdx.y, NB = gridDim.x;
  const int i_b = pid / Hkv, i_h = pid % Hkv;
  const long ub = ((long)i_b * NB + i_blk) * Hkv + i_h;
  const int n_iter = ucount[ub];
  const long tokb = (long)i_b * N_layer;
  const uint32_t sb = smem_u32(smem);
  const uint32_t bar0 = sb + NSLOT * SLOTB;
#if LMAX > 0
  for (int i = tid; i < n_iter; i += 32 * NWARP) {
    lsm[i] = ulist[ub * UMAX + i];
    lsm[LMAX + i] = umask[ub * UMAX + i];
  }
  const int* lp = lsm;
  const int* lm = lsm + LMAX;
#else
  const int* lp = ulist + ub * UMAX;
  const int* lm = umask + ub * UMAX;
#endif
  if (tid < NGRP) {
    mbar_init(bar0 + 8 * tid, USE_TMA ? 1 : 32);
    cnt[tid] = 0;
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  const int n_grp = (n_iter + EG - 1) / EG;
#if USE_TMA
  if (tid == 0)
    for (int g = 0; g < NGRP && g < n_grp; ++g)
      fill_group_tma(sb, g, g * EG, n_iter, lp, bar0 + 8 * g, &tmk, &tmv, (int)tokb, i_h);
#else
  if (w == 0)
    for (int g = 0; g < NGRP && g < n_grp; ++g)
      fill_group(sb, g, g * EG, n_iter, lp, bar0 + 8 * g, krot, vl, tokb, N_layer, Hkv, i_h, lane);
#endif

  // ---- per-thread rows: r0 = lane/4 (query j0 = r0/4, head g = r0%4) and r0 + 8 (query j0 + 2) of this warp ----
  const int r0 = lane >> 2, cc = (lane & 3) * 2, j0 = r0 >> 2, hg = r0 & 3;
  const int h = i_h * 4 + hg;
  int tq[2], rp[2], rc[2];
  bool rok[2];
  float a[2][8][4];
#if !C_SMEM
  float c16[4][4], s16[4][4];
#endif
#pragma unroll
  for (int ri = 0; ri < 2; ++ri) {
    int t = i_blk * BQB + 4 * w + j0 + 2 * ri;
    rok[ri] = t < T;
    t = min(t, T - 1);
    tq[ri] = t;
    const int tg = t_base + t;
    rp[ri] = tg / CRN;
    rc[ri] = tg % CRN;
    const long pq = posq[((long)i_b * T + t) * Hkv + i_h];
    const __nv_bfloat16* qp = q + (((long)i_b * T + t) * H + h) * 128;
#pragma unroll
    for (int kb = 0; kb < 4; ++kb)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int f = kb * 16 + cc + (i & 1) + (i >> 1) * 8;
        const float ql = rok[ri] ? __bfloat162float(qp[f]) : 0.f;
        const float qh = rok[ri] ? __bfloat162float(qp[f + 64]) : 0.f;
        const float cq = cosc[pq * 64 + f], sq = sinc[pq * 64 + f];
        if (ri == 0) {
          a[0][kb][i] = (ql * cq - qh * sq) * scale;
          a[0][kb + 4][i] = (ql * sq + qh * cq) * scale;
        } else {
          a[1][kb][i] = (ql * cq - qh * sq) * scale;
          a[1][kb + 4][i] = (ql * sq + qh * cq) * scale;
        }
      }
  }
#if C_SMEM
  if (tid < 64) {
    csm[tid] = cosc[CRN * 64 + tid];
    csm[64 + tid] = sinc[CRN * 64 + tid];
  }
  __syncthreads();
#else
#pragma unroll
  for (int kb = 0; kb < 4; ++kb)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int f = kb * 16 + cc + (i & 1) + (i >> 1) * 8;
      c16[kb][i] = cosc[CRN * 64 + f];
      s16[kb][i] = sinc[CRN * 64 + f];
    }
#endif
  // entries below every row's own parent (and inside the layer) need only the membership mask
  int pfast = min(rp[0], rp[1]);
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) pfast = min(pfast, __shfl_xor_sync(0xffffffffu, pfast, off));
  pfast = min(pfast, N_layer / CRN);

  float cmax[2] = {neg_inf, neg_inf}, csum[2] = {0.f, 0.f};
  float o[16][4];
#pragma unroll
  for (int nb = 0; nb < 16; ++nb)
#pragma unroll
    for (int i = 0; i < 4; ++i) o[nb][i] = 0.f;

  const int wsh = 4 * w;
  // ---- QK of one entry (scores of this warp's 16 rows vs the entry's 16 keys), one fp32 MMA chain per tile ----
  auto qk = [&](uint32_t kS, float (&sc)[2][4]) {
    float sh[2][4];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb)
#pragma unroll
      for (int i = 0; i < 4; ++i) sc[nb][i] = sh[nb][i] = 0.f;
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      const uint32_t A[4] = {pack_h2(a[0][kb][0], a[0][kb][1]), pack_h2(a[1][kb][0], a[1][kb][1]),
                             pack_h2(a[0][kb][2], a[0][kb][3]), pack_h2(a[1][kb][2], a[1][kb][3])};
      uint32_t b0, b1, b2, b3;
#if USE_TMA
      const int key = (lane & 7) + ((lane >> 4) << 3), ch = (kb & 3) * 2 + ((lane >> 3) & 1);
      ldsm_x4(kS + (kb >> 2) * 2048 + key * 128 + ((ch ^ (key & 7)) << 4), b0, b1, b2, b3);
#else
      ldsm_x4(kS + ((lane & 7) + ((lane >> 4) << 3)) * ROWB + kb * 32 + ((lane >> 3) & 1) * 16, b0, b1, b2, b3);
#endif
      if (ABL != 3 && SPLITK && kb >= 4) {
        mma16816(sh[0], A, b0, b1);
        mma16816(sh[1], A, b2, b3);
      } else if (ABL != 3) {
        mma16816(sc[0], A, b0, b1);
        mma16816(sc[1], A, b2, b3);
      } else {
        sc[0][kb & 3] += __uint_as_float(A[0] ^ b0);
        sc[1][kb & 3] += __uint_as_float(A[1] ^ b2);
      }
    }
    if (SPLITK) {
#pragma unroll
      for (int nb = 0; nb < 2; ++nb)
#pragma unroll
        for (int i = 0; i < 4; ++i) sc[nb][i] += sh[nb][i];
    }
  };
  // branch-free rotation of one row: rotate and keep by membership (no divergent branch around the FMAs)
  auto rot_sel = [&](float (&ar)[8][4], bool mem) {
#pragma unroll
    for (int kb = 0; kb < 4; ++kb)
#pragma unroll
      for (int h2 = 0; h2 < 2; ++h2) {
#if C_SMEM
        const int f = kb * 16 + cc + h2 * 8;
        const float2 c2 = *reinterpret_cast<const float2*>(csm + f);
        const float2 s2 = *reinterpret_cast<const float2*>(csm + 64 + f);
        const float cs[2] = {c2.x, c2.y}, sn[2] = {s2.x, s2.y};
#endif
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const int i = h2 * 2 + u;
#if C_SMEM
          const float cv = cs[u], sv = sn[u];
#else
          const float cv = c16[kb][i], sv = s16[kb][i];
#endif
          const float lo = ar[kb][i], hi = ar[kb + 4][i];
          const float nlo = lo * cv + hi * sv, nhi = hi * cv - lo * sv;
          ar[kb][i] = mem ? nlo : lo;
          ar[kb + 4][i] = mem ? nhi : hi;
        }
      }
  };
  // ---- mask, online softmax and PV of one entry whose scores are in sc ----
  auto finish = [&](uint32_t vS, int p, bool mem0, bool mem1, float (&sc)[2][4]) {
    float mx[2] = {-INFINITY, -INFINITY};
    if (FAST_MASK && p < pfast) {
#pragma unroll
      for (int nb = 0; nb < 2; ++nb)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const bool memr = (i >> 1) ? mem1 : mem0;
          sc[nb][i] = memr ? sc[nb][i] : -INFINITY;
          mx[i >> 1] = fmaxf(mx[i >> 1], sc[nb][i]);
        }
    } else {
#pragma unroll
      for (int nb = 0; nb < 2; ++nb)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int ri = i >> 1, key = nb * 8 + cc + (i & 1);
          const bool memr = ri ? mem1 : mem0;
          const bool v = memr && (p * CRN + key < N_layer) && (p != rp[ri] || key <= rc[ri]);
          sc[nb][i] = v ? sc[nb][i] : -INFINITY;
          mx[ri] = fmaxf(mx[ri], sc[nb][i]);
        }
    }
    float alpha[2];
#pragma unroll
    for (int ri = 0; ri < 2; ++ri) {
      mx[ri] = fmaxf(mx[ri], __shfl_xor_sync(0xffffffffu, mx[ri], 1));
      mx[ri] = fmaxf(mx[ri], __shfl_xor_sync(0xffffffffu, mx[ri], 2));
      const float nm = fmaxf(cmax[ri], mx[ri]);
      alpha[ri] = ex2s((cmax[ri] - nm) * LOG2E);
      cmax[ri] = nm;
    }
    float ps[2] = {0.f, 0.f};
#pragma unroll
    for (int nb = 0; nb < 2; ++nb)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int ri = i >> 1;
        sc[nb][i] = ABL == 4 ? (sc[nb][i] - cmax[ri]) : ex2s((sc[nb][i] - cmax[ri]) * LOG2E);
        ps[ri] += sc[nb][i];
      }
    csum[0] = csum[0] * alpha[0] + ps[0];
    csum[1] = csum[1] * alpha[1] + ps[1];
    if (!SKIP_RESCALE || __any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
      for (int nb = 0; nb < 16; ++nb) {
        o[nb][0] *= alpha[0];
        o[nb][1] *= alpha[0];
        o[nb][2] *= alpha[1];
        o[nb][3] *= alpha[1];
      }
    }
    const uint32_t P[4] = {pack_h2(sc[0][0], sc[0][1]), pack_h2(sc[0][2], sc[0][3]),
                           pack_h2(sc[1][0], sc[1][1]), pack_h2(sc[1][2], sc[1][3])};
#pragma unroll
    for (int nb = 0; nb < 16; nb += 2) {
      uint32_t b0, b1, b2, b3;
#if USE_TMA
      const int key = (lane & 7) + ((lane >> 3) & 1) * 8, d = nb + (lane >> 4);
      ldsm_x4_t(vS + (d >> 3) * 2048 + key * 128 + (((d & 7) ^ (key & 7)) << 4), b0, b1, b2, b3);
#else
      ldsm_x4_t(vS + ((lane & 7) + ((lane >> 3) & 1) * 8) * ROWB + (nb + (lane >> 4)) * 16, b0, b1, b2, b3);
#endif
      if (ABL != 2) {
        mma16816(o[nb], P, b0, b1);
        mma16816(o[nb + 1], P, b2, b3);
      } else {
        o[nb][0] += __uint_as_float(P[0] ^ b0 ^ b1);
        o[nb + 1][0] += __uint_as_float(P[1] ^ b2 ^ b3);
      }
    }
  };
  // ---- release a ring group; the last warp refills it with group g + NGRP ----
  auto release = [&](int g) {
    if (NO_WAIT) return;
    const int gs = g % NGRP;
    __syncwarp();
    int last = 0;
    if (lane == 0) last = atomicAdd(&cnt[gs], 1) == NWARP - 1;
    last = __shfl_sync(0xffffffffu, last, 0);
    if (last) {
      if (lane == 0) cnt[gs] = 0;
      const int gn = g + NGRP;
      if (gn < n_grp) {
#if USE_TMA
        if (lane == 0) {
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          fill_group_tma(sb, gs, gn * EG, n_iter, lp, bar0 + 8 * gs, &tmk, &tmv, (int)tokb, i_h);
        }
#else
        __syncwarp();
        fill_group(sb, gs, gn * EG, n_iter, lp, bar0 + 8 * gs, krot, vl, tokb, N_layer, Hkv, i_h, lane);
#endif
      }
    }
  };
#if USE_TMA
#define VOFF 4096
#else
#define VOFF TILEB
#endif
  // pending entry: its QK is done, its softmax + PV wait until the next member entry's QK has been issued
  int pend_g = -1, pend_p = 0;
  bool pend_m0 = false, pend_m1 = false;
  uint32_t pend_v = 0;
  float sp[2][4];
#if GL
  int4 m4n = make_int4(0, 0, 0, 0), p4n = make_int4(0, 0, 0, 0);
  if (n_grp > 0) {
    m4n = *reinterpret_cast<const int4*>(lm);
    p4n = *reinterpret_cast<const int4*>(lp);
  }
#endif
  for (int gi = 0; gi < n_grp; ++gi) {
    const int gs = gi % NGRP;
    if (LOOKAHEAD && pend_g >= 0 && pend_g + NGRP <= gi) {   // its slot is the one gi refills: finish it first
      finish(pend_v, pend_p, pend_m0, pend_m1, sp);
      release(pend_g);
      if (pend_m0) ROT(a[0]);
      if (pend_m1) ROT(a[1]);
      pend_g = -1;
    }
#if GL
    const int4 m4 = m4n, p4 = p4n;
    if (gi + 1 < n_grp) {                   // next group's masks / parents, in flight during this group
      m4n = *reinterpret_cast<const int4*>(lm + (gi + 1) * EG);
      p4n = *reinterpret_cast<const int4*>(lp + (gi + 1) * EG);
    }
    const int nv = n_iter - gi * EG;        // entries of this group that exist (>= 1)
    unsigned bits = (((m4.x >> wsh) & 15) ? 1u : 0u) | ((nv > 1 && ((m4.y >> wsh) & 15)) ? 2u : 0u) |
                    ((nv > 2 && ((m4.z >> wsh) & 15)) ? 4u : 0u) | ((nv > 3 && ((m4.w >> wsh) & 15)) ? 8u : 0u);
    if (ABL == 5) bits = 0;
#endif
    if (!NO_WAIT) mbar_wait(bar0 + 8 * gs, (gi / NGRP) & 1);
#pragma unroll 1
#if GL
    while (bits) {
      const int j = __ffs(bits) - 1;
      bits &= bits - 1;
      const int wm = ((j == 0 ? m4.x : j == 1 ? m4.y : j == 2 ? m4.z : m4.w) >> wsh) & 15;
      const int p = j == 0 ? p4.x : j == 1 ? p4.y : j == 2 ? p4.z : p4.w;
#else
    for (int j = 0; j < EG; ++j) {
      const int e = gi * EG + j;
      if (e >= n_iter) break;
      const int wm = (lm[e] >> wsh) & 15;
      if (!wm || ABL == 5) continue;
      const int p = lp[e];
#endif
      const bool mem0 = (wm >> j0) & 1, mem1 = (wm >> (j0 + 2)) & 1;
      const uint32_t kS = sb + (gs * EG + j) * SLOTB;
      if (LOOKAHEAD) {
        if (pend_g >= 0) {
          if (pend_m0) ROT(a[0]);
          if (pend_m1) ROT(a[1]);
          float sn[2][4];
          qk(kS, sn);
          finish(pend_v, pend_p, pend_m0, pend_m1, sp);
          if (pend_g != gi) release(pend_g);
#pragma unroll
          for (int nb = 0; nb < 2; ++nb)
#pragma unroll
            for (int i = 0; i < 4; ++i) sp[nb][i] = sn[nb][i];
        } else {
          qk(kS, sp);
        }
        pend_g = gi;
        pend_p = p;
        pend_m0 = mem0;
        pend_m1 = mem1;
        pend_v = kS + VOFF;
      } else {
        float sc[2][4];
        qk(kS, sc);
        if (ROT_EARLY) {
          if (ROT_SEL) {
            rot_sel(a[0], mem0);
            rot_sel(a[1], mem1);
          } else {
            if (mem0) ROT(a[0]);
            if (mem1) ROT(a[1]);
          }
        }
        finish(kS + VOFF, p, mem0, mem1, sc);
        if (!ROT_EARLY) {
          if (mem0) ROT(a[0]);
          if (mem1) ROT(a[1]);
        }
      }
    }
    if (!LOOKAHEAD || pend_g != gi) release(gi);
  }
  if (LOOKAHEAD && pend_g >= 0) {
    finish(pend_v, pend_p, pend_m0, pend_m1, sp);
    release(pend_g);
  }

#pragma unroll
  for (int ri = 0; ri < 2; ++ri) {
    float sm = csum[ri];
    sm += __shfl_xor_sync(0xffffffffu, sm, 1);
    sm += __shfl_xor_sync(0xffffffffu, sm, 2);
    if (rok[ri]) {
      const long st = ((long)i_b * T + tq[ri]) * H + h;
      if (FUSE) {
        // fused recsa_merge_to_global_kernel + recsa_final_normalize_kernel, same op order
        const float g_max = gmax[st], g_sum = gsum[st];
        const bool pc = sm > 1e-10f, pg = g_sum > 1e-10f;
        const float nm = fmaxf(g_max, cmax[ri]);
        const float sg = pg ? ex2f(__fmul_rn(__fsub_rn(g_max, nm), LOG2E)) : 0.f;
        const float sc = pc ? ex2f(__fmul_rn(__fsub_rn(cmax[ri], nm), LOG2E)) : 0.f;
        const float msum = __fmaf_rn(sg, g_sum, __fmul_rn(sc, sm));
        const float fsum = pc ? (pg ? msum : sm) : g_sum;
        const float* gp = gout + st * 128;
        __nv_bfloat16* op = fout + st * 128;
#pragma unroll
        for (int nb = 0; nb < 16; ++nb) {
          const float2 g2 = *reinterpret_cast<const float2*>(gp + nb * 8 + cc);
          float r[2];
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            const float ov = o[nb][2 * ri + u], gv = u ? g2.y : g2.x;
            const float mo = __fmaf_rn(sg, gv, __fmul_rn(sc, ov));
            const float fo = pc ? (pg ? mo : ov) : gv;
            float q;
            asm("div.full.f32 %0, %1, %2;" : "=f"(q) : "f"(fo), "f"(fsum));
            r[u] = q;
          }
          *reinterpret_cast<__nv_bfloat162*>(op + nb * 8 + cc) = __floats2bfloat162_rn(r[0], r[1]);
        }
      } else {
        if ((lane & 3) == 0) {
          out_max[st] = cmax[ri];
          out_sum[st] = sm;
        }
        float* op = out_o + st * 128;
#pragma unroll
        for (int nb = 0; nb < 16; ++nb)
          *reinterpret_cast<float2*>(op + nb * 8 + cc) = make_float2(o[nb][2 * ri], o[nb][2 * ri + 1]);
      }
    }
  }
}
