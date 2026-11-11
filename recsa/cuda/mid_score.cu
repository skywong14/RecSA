// RecSA mid-level selection scoring (union walk) for sm_90a.
#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda.h>

#ifndef NS
#define NS 8
#endif
#ifndef MINB
#define MINB 2
#endif
#ifndef ABL
#define ABL 0                   // timing ablations (wrong results): 1 one wgmma per entry, 2 no rotation / split, 3 no epilogue
#endif
#ifndef CSM
#define CSM 0                   // R(CR) coefficients from shared memory (frees 32 registers for 3 CTAs per SM)
#endif
#ifndef EARLY
#define EARLY 1                 // member rows' fp32 rotation while the entry's wgmma chain runs
#endif
#define BQ 16
#define CRN 16
#define TOPK 256
#define SLOTB 8192             // K hi lo | K hi hi | K res lo | K res hi: 16 tokens x 128 B each, 128B swizzle
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
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
#define DHI ((uint32_t)(1024 >> 4) | (1u << 30))
__device__ __forceinline__ uint32_t dlo(uint32_t addr) { return ((addr & 0x3FFFF) >> 4) | (1u << 16); }
// S (64 x 16) += A (registers) . B, B = 16 tokens x 16 dims K-major (128B swizzle)
template <int SD>
__device__ __forceinline__ void wg16(float (&d)[8], const uint32_t (&a)[4], uint32_t blo) {
  asm volatile(
      "{\n .reg .pred p;\n .reg .b64 db;\n setp.ne.b32 p, %14, 0;\n mov.b64 db, {%12, %13};\n"
      " wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
      "{%8, %9, %10, %11}, db, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(blo), "r"(DHI), "n"(SD));
}
__device__ __forceinline__ uint32_t h2u(__half lo, __half hi) {
  return (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
}

extern "C" __global__ void __launch_bounds__(128, MINB)
mid_score_cuda(const __nv_bfloat16* __restrict__ q, const int* __restrict__ ulist, const int* __restrict__ umask,
               const int* __restrict__ ucount, const int* __restrict__ posq, const float* __restrict__ cosc,
               const float* __restrict__ sinc, float* __restrict__ raw, float* __restrict__ pmax,
               int t_offset, int t_base, int n_prog_t, int T, int N_layer, int UMAX, int layer_power, int H, int Hkv,
               int Bn, float scale, float neg_inf, const __grid_constant__ CUtensorMap tmk) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
  const uint32_t sb = smem_u32(smem);
  const uint32_t bar0 = sb + NS * SLOTB;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int i_blk = blockIdx.x, pid = blockIdx.y, NB = gridDim.x;
  const int i_b = pid / Hkv, i_h = pid % Hkv;
  const long ub = ((long)i_b * NB + i_blk) * Hkv + i_h;
  const int n_iter = ucount[ub];
  const int* lp = ulist + ub * UMAX;
  const int* lm = umask + ub * UMAX;
  const int rowk0 = i_b * N_layer, rowr0 = Bn * N_layer + i_b * N_layer;
  auto fill = [&](int e) {          // one thread: the 4 boxes of entry e into slot e % NS
    const int s = e % NS;
    const uint32_t st = sb + s * SLOTB, bar = bar0 + 8 * s;
    const int tok = lp[e] * CRN;
    mbar_expect_tx(bar, SLOTB);
    tma3(st, &tmk, 0, i_h, rowk0 + tok, bar);
    tma3(st + 2048, &tmk, 64, i_h, rowk0 + tok, bar);
    tma3(st + 4096, &tmk, 0, i_h, rowr0 + tok, bar);
    tma3(st + 6144, &tmk, 64, i_h, rowr0 + tok, bar);
  };
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) mbar_init(bar0 + 8 * s, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    for (int e = 0; e < NS && e < n_iter; ++e) fill(e);
  }
  // ---- this thread's rows: x = 0: 16 w + lane / 4 (query qa = 4 w + lane / 16, head hd), x = 1: + 8 (query + 2) ----
  const int hd = (lane >> 2) & 3, t4 = lane & 3;
  int qx[2], rpx[2], rcx[2];
  bool okx[2];
  float al[2][16], ah[2][16];       // a_lo / a_hi of the row at dims f = 16 kk + 8 hh + 2 t4 + u (index 4 kk + 2 hh + u)
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    const int qi = 4 * w + (lane >> 4) + 2 * x;
    qx[x] = qi;
    int pl = i_blk * BQ + qi;
    okx[x] = pl < n_prog_t;
    pl = min(pl, n_prog_t - 1);
    const int t_loc = t_offset + pl;
    const int rm = (t_base + t_loc) / layer_power;
    rpx[x] = rm / CRN;
    rcx[x] = rm % CRN;
    const long pq = posq[((long)i_b * T + t_loc) * Hkv + i_h];
    const __nv_bfloat16* qp = q + (((long)i_b * T + t_loc) * H + (i_h * 4 + hd)) * 128;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int f = 16 * kk + 8 * hh + 2 * t4;
        const __nv_bfloat162 q2l = okx[x] ? *reinterpret_cast<const __nv_bfloat162*>(qp + f) : __floats2bfloat162_rn(0.f, 0.f);
        const __nv_bfloat162 q2h = okx[x] ? *reinterpret_cast<const __nv_bfloat162*>(qp + f + 64) : __floats2bfloat162_rn(0.f, 0.f);
        const float2 c2 = *reinterpret_cast<const float2*>(cosc + pq * 64 + f);
        const float2 s2 = *reinterpret_cast<const float2*>(sinc + pq * 64 + f);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const float ql = __bfloat162float(u ? q2l.y : q2l.x), qh = __bfloat162float(u ? q2h.y : q2h.x);
          const float c = u ? c2.y : c2.x, sn = u ? s2.y : s2.x;
          al[x][4 * kk + 2 * hh + u] = __fmul_rn(scale, __fmaf_rn(ql, c, -__fmul_rn(qh, sn)));
          ah[x][4 * kk + 2 * hh + u] = __fmul_rn(scale, __fmaf_rn(qh, c, __fmul_rn(ql, sn)));
        }
      }
  }
  // R(CR) coefficients of this thread's 16 dims (shared by both rows)
#if CSM
  float* csm = reinterpret_cast<float*>(smem + NS * SLOTB + 8 * NS + 64);      // [cos 64 | sin 64]
  if (tid < 64) {
    csm[tid] = cosc[CRN * 64 + tid];
    csm[64 + tid] = sinc[CRN * 64 + tid];
  }
#define C16(i) csm[16 * ((i) >> 2) + 8 * (((i) >> 1) & 1) + 2 * t4 + ((i) & 1)]
#define S16(i) csm[64 + 16 * ((i) >> 2) + 8 * (((i) >> 1) & 1) + 2 * t4 + ((i) & 1)]
#else
  float c16[16], s16[16];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int f = 16 * kk + 8 * hh + 2 * t4 + u;
        c16[4 * kk + 2 * hh + u] = cosc[CRN * 64 + f];
        s16[4 * kk + 2 * hh + u] = sinc[CRN * 64 + f];
      }
#define C16(i) c16[i]
#define S16(i) s16[i]
#endif
  // fp16 fragments: F[m][kk][x + 2 hh] = pack(dims f, f + 1) of row x; m: 0 al_h, 1 ah_h, 2 al_r, 3 ah_r
  uint32_t F[4][4][4];
  auto split = [&](int x) {
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int i = 4 * kk + 2 * hh;
        const __half l0 = __float2half_rn(al[x][i]), l1 = __float2half_rn(al[x][i + 1]);
        const __half h0 = __float2half_rn(ah[x][i]), h1 = __float2half_rn(ah[x][i + 1]);
        F[0][kk][x + 2 * hh] = h2u(l0, l1);
        F[1][kk][x + 2 * hh] = h2u(h0, h1);
        F[2][kk][x + 2 * hh] = h2u(__float2half_rn(__fsub_rn(al[x][i], __half2float(l0))),
                                   __float2half_rn(__fsub_rn(al[x][i + 1], __half2float(l1))));
        F[3][kk][x + 2 * hh] = h2u(__float2half_rn(__fsub_rn(ah[x][i], __half2float(h0))),
                                   __float2half_rn(__fsub_rn(ah[x][i + 1], __half2float(h1))));
      }
  };
  auto rotate = [&](int x) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float lo = al[x][i], hi = ah[x][i];
      const float c = C16(i), sn = S16(i);
      al[x][i] = __fmaf_rn(lo, c, __fmul_rn(hi, sn));
      ah[x][i] = __fmaf_rn(hi, c, -__fmul_rn(lo, sn));
    }
  };
  split(0);
  split(1);
  __syncthreads();

  // output rows of this lane's queries (lanes with hd == 0 of each query store)
  long rbase[2], mbase[2];
  int ptr[2] = {0, 0};
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    const int pl = min(i_blk * BQ + qx[x], n_prog_t - 1);
    rbase[x] = ((long)pid * n_prog_t + pl) * (TOPK * CRN);
    mbase[x] = ((long)pid * n_prog_t + pl) * TOPK;
  }
  const int cb = 2 * t4;
  for (int it = 0; it < n_iter; ++it) {
    const int s = it % NS;
    const uint32_t st = sb + s * SLOTB;
    const int p = lp[it], m = lm[it];
    bool mem[2];
#pragma unroll
    for (int x = 0; x < 2; ++x) mem[x] = okx[x] && ((m >> qx[x]) & 1);
    mbar_wait(bar0 + 8 * s, (it / NS) & 1);
    float d[8];
    {
      const uint32_t kh = dlo(st), kr = dlo(st + 4096);
      wg_fence();
      if (ABL == 1) {
        wg16<0>(d, F[0][0], kh);
        wg_commit();
        wg_wait<0>();
        goto chain_done;
      }
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {        // al_h . K_lo (dims 0..63 = box 0)
        if (kk == 0) wg16<0>(d, F[0][kk], kh + kk * 2);
        else wg16<1>(d, F[0][kk], kh + kk * 2);
      }
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) wg16<1>(d, F[1][kk], kh + 128 + kk * 2);   // ah_h . K_hi (box 1, +2048 B)
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) wg16<1>(d, F[0][kk], kr + kk * 2);         // al_h . R_lo
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) wg16<1>(d, F[1][kk], kr + 128 + kk * 2);   // ah_h . R_hi
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) wg16<1>(d, F[2][kk], kh + kk * 2);         // al_r . K_lo
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) wg16<1>(d, F[3][kk], kh + 128 + kk * 2);   // ah_r . K_hi
      wg_commit();
#if EARLY && ABL != 2
#pragma unroll
      for (int x = 0; x < 2; ++x)
        if (mem[x]) rotate(x);
#endif
      wg_wait<0>();
    }
  chain_done:
    // the slot is free: refill with entry it + NS
    if (it + NS < n_iter) {
      asm volatile("bar.sync 1, 128;" ::: "memory");
      if (tid == 0) fill(it + NS);
    }
    if (ABL == 3) {
      if (d[0] == 12345.f) raw[0] = d[1];
#pragma unroll
      for (int x = 0; x < 2; ++x) ptr[x] += mem[x];
      goto epi_done;
    }
    {
    // ---- mask, max over children and heads, exp, head sum (d[4 i + c]: child 8 i + cb + (c & 1), row x = c >> 1) ----
    float v[8];
    bool val[8];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int x = c >> 1, ch = 8 * i + cb + (c & 1), tok = p * CRN + ch;
        const bool ok = mem[x] && tok < N_layer && (p != rpx[x] || ch <= rcx[x]);
        val[4 * i + c] = ok;
        v[4 * i + c] = ok ? d[4 * i + c] : neg_inf;
      }
    float mx[2];
#pragma unroll
    for (int x = 0; x < 2; ++x) {
      float r = fmaxf(fmaxf(v[2 * x], v[2 * x + 1]), fmaxf(v[4 + 2 * x], v[4 + 2 * x + 1]));
      r = fmaxf(r, __shfl_xor_sync(0xffffffffu, r, 1));
      r = fmaxf(r, __shfl_xor_sync(0xffffffffu, r, 2));
      r = fmaxf(r, __shfl_xor_sync(0xffffffffu, r, 4));
      r = fmaxf(r, __shfl_xor_sync(0xffffffffu, r, 8));
      mx[x] = r;
    }
    float e[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float t = ex2f(__fmul_rn(__fsub_rn(v[k], mx[(k >> 1) & 1]), LOG2E_F));
      e[k] = val[k] ? t : 0.f;
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float t = e[k] + __shfl_xor_sync(0xffffffffu, e[k], 8);
      t = t + __shfl_xor_sync(0xffffffffu, t, 4);
      e[k] = val[k] ? t : neg_inf;          // vq: every head of the query shares the row validity
    }
    if (hd == 0) {
#pragma unroll
      for (int x = 0; x < 2; ++x)
        if (mem[x]) {
          float* rp = raw + rbase[x] + ptr[x] * CRN + cb;
          *reinterpret_cast<float2*>(rp) = make_float2(e[2 * x], e[2 * x + 1]);
          *reinterpret_cast<float2*>(rp + 8) = make_float2(e[4 + 2 * x], e[4 + 2 * x + 1]);
          if (t4 == 0) pmax[mbase[x] + ptr[x]] = mx[x];
        }
    }
#pragma unroll
    for (int x = 0; x < 2; ++x) ptr[x] += mem[x];
    }
  epi_done:
    if (ABL == 2) continue;
    // ---- member rows: q rotated by R(-CR), fp16 split refreshed (only slots with a member in the warp) ----
#pragma unroll
    for (int x = 0; x < 2; ++x) {
      if (__any_sync(0xffffffffu, mem[x])) {
#if !EARLY
        if (mem[x]) rotate(x);
#endif
        split(x);
      }
    }
  }
}
