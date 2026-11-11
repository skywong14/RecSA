// Top-level selection finalize, one warp per (query, kv head) row, sm_90a.
#include <stdint.h>

#define LOG2E_BITS 0x3FB8AA3Bu
#ifndef NR16_BLOCKS
#define NR16_BLOCKS 2
#endif
#ifndef CARRY
#define CARRY 1                 // keys in the unsigned order, counts as carry-outs
#endif
#ifndef ESKIP
#define ESKIP 0                 // emission: skip element rows without a kept element
#endif
#ifndef PFAC
#define PFAC 1                  // mid finalize: one exp per parent (owner lane), shuffled to its 16 slots
#endif
#ifndef EMIT2
#define EMIT2 1                 // emission by walking the set bits of the bitmask words
#endif
// exclusive warp prefix of c (lane order) and the warp total
__device__ __forceinline__ int warp_excl(int c, int lane, int& total) {
  int incl = c;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, off);
    if (lane >= off) incl += y;
  }
  total = __shfl_sync(0xffffffffu, incl, 31);
  return incl - c;
}
#if CARRY
typedef uint32_t skey_t;
#define KEY_OF(enc) ((uint32_t)((enc) ^ (((enc) >> 31) | (int)0x80000000)))
#define THR_OF(cand) (cand)
#else
typedef int skey_t;
#define KEY_OF(enc) (((enc) ^ (((enc) >> 31) | (int)0x80000000)) ^ (int)0x80000000)
#define THR_OF(cand) ((int)((cand) ^ 0x80000000u))
#endif
// acc + (u >= cand) in the unsigned order: the carry-out of u - cand
__device__ __forceinline__ uint32_t cnt_ge(uint32_t acc, uint32_t u, uint32_t cand) {
  uint32_t r;
  asm("{\n .reg .u32 t;\n sub.cc.u32 t, %1, %2;\n addc.u32 %0, %3, 0;\n}" : "=r"(r) : "r"(u), "r"(cand), "r"(acc));
  return r;
}

__device__ __forceinline__ float ex2f(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// tl.exp(a - b) as Triton lowers it: sub.f32, mul.f32 by log2(e), ex2.approx.f32
__device__ __forceinline__ float texp_diff(float a, float b) {
  return ex2f(__fmul_rn(__fsub_rn(a, b), __uint_as_float(LOG2E_BITS)));
}

template <int NR>
__device__ __forceinline__ void finalize_row(const float* __restrict__ raw, const float* __restrict__ pmax,
                                             int* __restrict__ next_sel, int* __restrict__ bitmask, int t_base, int q0,
                                             int T, int QC, int NP, int NT, int NWORDS, int lq, int pid,
                                             int layer_power, int H_kv, int TOP_K, int LOG_N, float SVT, float NEG_INF) {
  constexpr int E = 8 * NR;      // elements per lane
  constexpr int NTI = 4 * NR;    // 64-node tiles
  const int lane = threadIdx.x & 31;
  const int i_b = pid / H_kv, i_h = pid % H_kv;
  const int t_loc = q0 + lq;
  const int ncand = (t_base + t_loc) / layer_power + 1;
  const int n_batches = (ncand + 255) / 256;
  const long rrow = (long)pid * QC + lq;
  const float* rp_raw = raw + rrow * NP;
  const float* rp_pm = pmax + rrow * NT;
  const int idx_mask = (1 << LOG_N) - 1;

  // tile maxima (warp-uniform) and batch maxima over live tiles
  float tm[NTI];
#pragma unroll
  for (int t = 0; t < NTI; ++t) tm[t] = (64 * t < ncand) ? rp_pm[t] : NEG_INF;
  float bm[NR];
#pragma unroll
  for (int b = 0; b < NR; ++b)
    bm[b] = fmaxf(fmaxf(tm[4 * b], tm[4 * b + 1]), fmaxf(tm[4 * b + 2], tm[4 * b + 3]));
  // batch-relative factor per tile: exp(m_tile - bm)
  float ft[NTI];
#pragma unroll
  for (int t = 0; t < NTI; ++t) ft[t] = texp_diff(tm[t], bm[t / 4]);
  float rb[NR], rpf[NR];
  {
    float run = NEG_INF, prv = NEG_INF;
#pragma unroll
    for (int b = 0; b < NR; ++b) {
      prv = b == 0 ? NEG_INF : run;     // max over batches < b
      run = fmaxf(run, bm[b]);          // max over batches <= b
      rb[b] = texp_diff(bm[b], run);
      rpf[b] = texp_diff(prv, run);
    }
  }
  float v[E];
#pragma unroll
  for (int k = 0; k < E; ++k) {
    const int e = 32 * k + lane;
    const bool live = e < ncand;
    float x = live ? rp_raw[e] : NEG_INF;
    x = x >= 0.f ? __fmul_rn(x, ft[k / 2]) : x;
    x = x >= 0.f ? __fmul_rn(x, rb[k / 8]) : x;
    x = (e == ncand - 1) ? 1e6f : x;
    x = live ? x : NEG_INF;
    v[k] = __int_as_float(__float_as_int(x) & ~idx_mask);
  }
  // earlier batches re-scaled by every later running-max raise and re-truncated
#pragma unroll
  for (int j = 1; j < NR; ++j) {
    const float f = rpf[j];
    if (f != 1.f && j < n_batches) {
#pragma unroll
      for (int k = 0; k < 8 * j; ++k) {
        const float x = v[k] >= 0.f ? __fmul_rn(v[k], f) : v[k];
        v[k] = __int_as_float(__float_as_int(x) & ~idx_mask);
      }
    }
  }
  // encode; remember (v > SVT) & live per element
  skey_t us[E];
  uint32_t okb[(E + 31) / 32];
#pragma unroll
  for (int w = 0; w < (E + 31) / 32; ++w) okb[w] = 0;
#pragma unroll
  for (int k = 0; k < E; ++k) {
    const int e = 32 * k + lane;
    const int vi = __float_as_int(v[k]);
    const int enc_idx = (v[k] >= 0.f ? ~e : e) & idx_mask;
    const int enc = (vi & ~idx_mask) | enc_idx;
    us[k] = KEY_OF(enc);
    if (v[k] > SVT && e < ncand) okb[k >> 5] |= 1u << (k & 31);
  }
  // TOP_K-th largest key: bit bisection in the unsigned order, stop at the first candidate with count == TOP_K
  skey_t taus = THR_OF(0u);
  if (ncand > TOP_K) {
    uint32_t tau = 0;
#pragma unroll 1
    for (int step = 0; step < 32; ++step) {
      const uint32_t cand = tau | (1u << (31 - step));
#if CARRY
      uint32_t cp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int k = 0; k < E; ++k) cp[k & 7] = cnt_ge(cp[k & 7], us[k], cand);
      int c = (int)(((cp[0] + cp[1]) + (cp[2] + cp[3])) + ((cp[4] + cp[5]) + (cp[6] + cp[7])));
#else
      const int thr = (int)(cand ^ 0x80000000u);
      int cp[8] = {0, 0, 0, 0, 0, 0, 0, 0};      // independent partial counts (short add chains)
#pragma unroll
      for (int k = 0; k < E; ++k) cp[k & 7] += us[k] >= thr;
      int c = ((cp[0] + cp[1]) + (cp[2] + cp[3])) + ((cp[4] + cp[5]) + (cp[6] + cp[7]));
#endif
      c = __reduce_add_sync(0xffffffffu, c);
      if (c >= TOP_K) tau = cand;
      if (c == TOP_K) break;
    }
    taus = THR_OF(tau);
  }
  // emit kept nodes in node order + bitmask words (word k = ballot of element row k)
  const long obase = (((long)i_b * T + t_loc) * H_kv + i_h) * TOP_K;
  const long wbase = (((long)i_b * T + t_loc) * H_kv + i_h) * NWORDS;
  const uint32_t lt = (1u << lane) - 1u;
  int base = 0;
  uint32_t myw[(E + 31) / 32];          // lane l keeps bitmask words k = l, l + 32, ...
#if EMIT2
#pragma unroll
  for (int w = 0; w < (E + 31) / 32; ++w) myw[w] = 0u;
#pragma unroll
  for (int k = 0; k < E; ++k) {
    const bool kp = us[k] >= taus && ((okb[k >> 5] >> (k & 31)) & 1u);
    const uint32_t bal = __ballot_sync(0xffffffffu, kp);
    if ((k & 31) == lane) myw[k >> 5] = bal;
  }
  // node order = word order (k = 32 w + lane), then bit order
#pragma unroll
  for (int w = 0; w < (E + 31) / 32; ++w) {
    int tot;
    int r = base + warp_excl(__popc(myw[w]), lane, tot);
    base += tot;
    uint32_t bits = myw[w];
    const int e0 = 32 * (32 * w + lane);
    while (bits) {
      const int b = __ffs(bits) - 1;
      bits &= bits - 1u;
      if (r < TOP_K) next_sel[obase + r] = e0 + b;
      ++r;
    }
  }
#else
#pragma unroll
  for (int k = 0; k < E; ++k) {
    const bool kp = us[k] >= taus && ((okb[k >> 5] >> (k & 31)) & 1u);
    const uint32_t bal = __ballot_sync(0xffffffffu, kp);
    if ((k & 31) == lane) myw[k >> 5] = bal;
    if (!ESKIP || bal) {
      const int rk = base + __popc(bal & lt);
      int* dst = next_sel + obase + rk;
      asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %0, 0;\n @p st.global.b32 [%1], %2;\n}\n"
                   :: "r"((int)(kp && rk < TOP_K)), "l"(dst), "r"(32 * k + lane) : "memory");
      base += __popc(bal);
    }
  }
#endif
#pragma unroll
  for (int w = 0; w < (E + 31) / 32; ++w) {
    const int k = 32 * w + lane;
    if (k < E && k < NWORDS) bitmask[wbase + k] = (int)myw[w];
  }
}

#define FINALIZE_KERNEL(NRV)                                                                                         \
  extern "C" __global__ void __launch_bounds__(128, NRV >= 12 ? NR16_BLOCKS : 4) top_finalize_nr##NRV(                       \
      const float* __restrict__ raw, const float* __restrict__ pmax, int* __restrict__ next_sel,                      \
      int* __restrict__ bitmask, int t_base, int q0, int T, int QC, int NP, int NT, int NWORDS, int lq0, int nq,     \
      int layer_power, int H_kv, int TOP_K, int LOG_N, float SVT, float NEG_INF) {                                     \
    const int lq_rel = blockIdx.x * 4 + (threadIdx.x >> 5);                                                         \
    if (lq_rel >= nq) return;                                                                                          \
    finalize_row<NRV>(raw, pmax, next_sel, bitmask, t_base, q0, T, QC, NP, NT, NWORDS, lq0 + lq_rel, blockIdx.y,     \
                      layer_power, H_kv, TOP_K, LOG_N, SVT, NEG_INF);                                                 \
  }
FINALIZE_KERNEL(1)
FINALIZE_KERNEL(2)
FINALIZE_KERNEL(3)
FINALIZE_KERNEL(4)
FINALIZE_KERNEL(6)
FINALIZE_KERNEL(8)
FINALIZE_KERNEL(12)
FINALIZE_KERNEL(16)

// ----------------------------------------------------------------------------------------------------------------
// Mid-level finalize (mid_select_finalize_kernel, STORE_SCRATCH = 0), one warp per (query, kv head) row:
// candidates are the CR children of the row's TOP_K (= 256) listed parents, slot = parent rank * CR + child, so the
// row has 4096 slots in 16 batches of 256 (16 parents each). Children are emitted in slot order as node ids.
// NR (4 / 8 / 12 / 16 batches): the row lists at most 16 NR parents.
template <int NR>
__device__ __forceinline__ void mid_finalize_row(
    const float* __restrict__ raw, const float* __restrict__ pmax, const int* __restrict__ psel,
    int* __restrict__ nsel, int t_offset, int t_base, int n_prog_t, int nq, int T, int layer_power, int H_kv,
    int LOG_N, float SVT, float NEG_INF) {
  constexpr int E = 8 * NR, CRN = 16, TOP_K = 256, NU = NR / 2;   // NU: 32-rank parent words in use
  const int lane = threadIdx.x & 31;
  const int pid_local = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (pid_local >= nq) return;
  const int pid = blockIdx.y;
  const int i_b = pid / H_kv, i_h = pid % H_kv;
  const int t_loc = t_offset + pid_local;
  const int rc = ((t_base + t_loc) / layer_power) % CRN;
  const long prow = (((long)i_b * T + t_loc) * H_kv + i_h) * TOP_K;
  // number of listed parents (the list is ascending with the empty slots (-1) last)
  int pr[TOP_K / 32];                          // listed parent ids: rank p = 32 u + lane
  int nvp = 0;
#pragma unroll
  for (int u = 0; u < TOP_K / 32; ++u) {
    pr[u] = psel[prow + 32 * u + lane];
    nvp += pr[u] >= 0;
  }
  nvp = __reduce_add_sync(0xffffffffu, nvp);
  const int ncand = (nvp - 1) * CRN + rc + 1;
  const int n_batches = (nvp + 15) / 16;
  const long rr = (long)pid * n_prog_t + pid_local;
  const float* rp_raw = raw + rr * (TOP_K * CRN);
  const float* rp_pm = pmax + rr * TOP_K;
  const int idx_mask = (1 << LOG_N) - 1;
  // parent-local maxima (rank p = 32 u + lane) and batch maxima over the live parents of each batch (16 ranks = one
  // half warp of one u): max over the half warp, broadcast
  float pm[NU];
#pragma unroll
  for (int u = 0; u < NU; ++u) pm[u] = (CRN * (32 * u + lane) < ncand) ? rp_pm[32 * u + lane] : NEG_INF;
  float bm[NR];
#pragma unroll
  for (int u = 0; u < NU; ++u) {
    float m = pm[u];
#pragma unroll
    for (int off = 8; off >= 1; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    bm[2 * u] = __shfl_sync(0xffffffffu, m, 0);
    bm[2 * u + 1] = __shfl_sync(0xffffffffu, m, 16);
  }
  float rb[NR], rpf[NR];
  {
    float run = NEG_INF, prv = NEG_INF;
#pragma unroll
    for (int b = 0; b < NR; ++b) {
      prv = b == 0 ? NEG_INF : run;
      run = fmaxf(run, bm[b]);
      rb[b] = texp_diff(bm[b], run);
      rpf[b] = texp_diff(prv, run);
    }
  }
#if PFAC
  // factor of parent rank 32 u + lane (batch (32 u + lane) / 16 = 2 u + lane / 16); a live slot's parent is live
  float fpar[NU];
#pragma unroll
  for (int u = 0; u < NU; ++u) fpar[u] = texp_diff(pm[u], (lane & 16) ? bm[2 * u + 1] : bm[2 * u]);
#endif
  float v[E];
#pragma unroll
  for (int k = 0; k < E; ++k) {
    const int e = 32 * k + lane;               // slot; parent rank e / 16 = 2k + lane / 16
    const bool live = e < ncand;
    float x = live ? rp_raw[e] : NEG_INF;
#if PFAC
    const float fp = __shfl_sync(0xffffffffu, fpar[k / 16], (2 * k + (lane >> 4)) & 31);
    x = x >= 0.f ? __fmul_rn(x, fp) : x;
#else
    const float mps = __shfl_sync(0xffffffffu, pm[k / 16], (2 * k + (lane >> 4)) & 31);   // all lanes
    const float mp = live ? mps : NEG_INF;
    x = x >= 0.f ? __fmul_rn(x, texp_diff(mp, bm[k / 8])) : x;
#endif
    x = x >= 0.f ? __fmul_rn(x, rb[k / 8]) : x;
    x = (e == ncand - 1) ? 1e6f : x;
    x = live ? x : NEG_INF;
    v[k] = __int_as_float(__float_as_int(x) & ~idx_mask);
  }
#pragma unroll
  for (int j = 1; j < NR; ++j) {
    const float f = rpf[j];
    if (f != 1.f && j < n_batches) {
#pragma unroll
      for (int k = 0; k < 8 * j; ++k) {
        const float x = v[k] >= 0.f ? __fmul_rn(v[k], f) : v[k];
        v[k] = __int_as_float(__float_as_int(x) & ~idx_mask);
      }
    }
  }
  skey_t us[E];
  uint32_t okb[E / 32];
#pragma unroll
  for (int w = 0; w < E / 32; ++w) okb[w] = 0;
#pragma unroll
  for (int k = 0; k < E; ++k) {
    const int e = 32 * k + lane;
    const int vi = __float_as_int(v[k]);
    const int enc_idx = (v[k] >= 0.f ? ~e : e) & idx_mask;
    const int enc = (vi & ~idx_mask) | enc_idx;
    us[k] = KEY_OF(enc);
    if (v[k] > SVT && e < ncand) okb[k >> 5] |= 1u << (k & 31);
  }
  uint32_t tau = 0;
#pragma unroll 1
  for (int step = 0; step < 32; ++step) {
    const uint32_t cand = tau | (1u << (31 - step));
#if CARRY
    uint32_t cp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < E; ++k) cp[k & 7] = cnt_ge(cp[k & 7], us[k], cand);
    int c = (int)(((cp[0] + cp[1]) + (cp[2] + cp[3])) + ((cp[4] + cp[5]) + (cp[6] + cp[7])));
#else
    const int thr = (int)(cand ^ 0x80000000u);
    int cp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < E; ++k) cp[k & 7] += us[k] >= thr;
    int c = ((cp[0] + cp[1]) + (cp[2] + cp[3])) + ((cp[4] + cp[5]) + (cp[6] + cp[7]));
#endif
    c = __reduce_add_sync(0xffffffffu, c);
    if (c >= TOP_K) tau = cand;
    if (c == TOP_K) break;
  }
  const skey_t taus = THR_OF(tau);
  const uint32_t lt = (1u << lane) - 1u;
  int base = 0;
#if EMIT2
  __shared__ int spr_all[4][TOP_K];
  int* spr = spr_all[(threadIdx.x >> 5) & 3];
#pragma unroll
  for (int u = 0; u < TOP_K / 32; ++u) spr[32 * u + lane] = pr[u];
  uint32_t myw[E / 32];
#pragma unroll
  for (int k = 0; k < E; ++k) {
    const bool kp = us[k] >= taus && ((okb[k >> 5] >> (k & 31)) & 1u);
    const int par = __shfl_sync(0xffffffffu, pr[k / 16], (2 * k + (lane >> 4)) & 31);
    const uint32_t bal = __ballot_sync(0xffffffffu, kp && par >= 0);
    if ((k & 31) == lane) myw[k >> 5] = bal;
  }
  __syncwarp();
#pragma unroll
  for (int w = 0; w < E / 32; ++w) {
    int tot;
    int r = base + warp_excl(__popc(myw[w]), lane, tot);
    base += tot;
    uint32_t bits = myw[w];
    const int e0 = 32 * (32 * w + lane);
    while (bits) {
      const int b = __ffs(bits) - 1;
      bits &= bits - 1u;
      const int e = e0 + b;
      if (r < TOP_K) nsel[prow + r] = spr[e >> 4] * CRN + (e & 15);
      ++r;
    }
  }
  (void)lt;
#else
#pragma unroll
  for (int k = 0; k < E; ++k) {
    const int e = 32 * k + lane;
    const bool kp = us[k] >= taus && ((okb[k >> 5] >> (k & 31)) & 1u);
    const int par = __shfl_sync(0xffffffffu, pr[k / 16], (2 * k + (lane >> 4)) & 31);
    const bool ok = kp && par >= 0;
    const uint32_t bal = __ballot_sync(0xffffffffu, ok);
    const int rk = base + __popc(bal & lt);
    int* dst = nsel + prow + rk;
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %0, 0;\n @p st.global.b32 [%1], %2;\n}\n"
                 :: "r"((int)(ok && rk < TOP_K)), "l"(dst), "r"(par * CRN + (e & 15)) : "memory");
    base += __popc(bal);
  }
#endif
}

#define MID_FINALIZE_KERNEL(NAME, NRV, MB)                                                                            \
  extern "C" __global__ void __launch_bounds__(128, MB) NAME(                                                         \
      const float* __restrict__ raw, const float* __restrict__ pmax, const int* __restrict__ psel,                    \
      int* __restrict__ nsel, int t_offset, int t_base, int n_prog_t, int nq, int T, int layer_power, int H_kv,      \
      int LOG_N, float SVT, float NEG_INF) {                                                                          \
    mid_finalize_row<NRV>(raw, pmax, psel, nsel, t_offset, t_base, n_prog_t, nq, T, layer_power, H_kv, LOG_N, SVT,   \
                          NEG_INF);                                                                                    \
  }
MID_FINALIZE_KERNEL(mid_finalize16, 16, 2)
MID_FINALIZE_KERNEL(mid_finalize_nr12, 12, 2)
MID_FINALIZE_KERNEL(mid_finalize_nr8, 8, 4)
MID_FINALIZE_KERNEL(mid_finalize_nr4, 4, 4)
