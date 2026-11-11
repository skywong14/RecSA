// Prefill top accumulate (recsa_top_accumulate_gqa_kernel, PREROT / fp16 dots) as a hand-written sm_90a kernel.
#include <stdint.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define TN 64                        // nodes per tile
#ifndef NSTAGE
#define NSTAGE 2
#endif
#ifndef MINB
#define MINB 3
#endif
#ifndef MASK2
#define MASK2 1                      // pre-shifted bitmask words, causal limit relative to the lane's keys
#endif
#ifndef WG
#define WG 0                         // QK and PV as wgmma (RS form, the same k16 chains as the mma.sync path)
#endif
#define STAGEB (4 * 64 * 128)        // K lo | K hi | V lo | V hi boxes: 64 rows x 128 B each
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
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma3(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               :: "r"(dst), "l"((uint64_t)map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
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
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
// 128B-swizzle descriptors: K-major (keys: LBO unused, SBO 1024) and MN-major (values^T: LBO 8192 between the two
// 64-dim boxes, SBO 1024 between 8-node groups); lo word = start >> 4 | LBO, hi word = SBO | swizzle 128B
__device__ __forceinline__ uint32_t dlo(uint32_t addr, uint32_t lbo) { return ((addr & 0x3FFFF) >> 4) | ((lbo >> 4) << 16); }
#define DHI ((uint32_t)(1024 >> 4) | (1u << 30))
// S (64 x 64) += A (registers) . K^T
template <int SD>
__device__ __forceinline__ void wg_qk(float (&d)[8][4], const uint32_t (&a)[4], uint32_t blo) {
  asm volatile(
      "{\n .reg .pred p;\n .reg .b64 db;\n setp.ne.b32 p, %38, 0;\n mov.b64 db, {%36, %37};\n"
      " wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "{%32, %33, %34, %35}, db, p, 1, 1, 0;\n}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
        "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]),
        "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]),
        "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(blo), "r"(DHI), "n"(SD));
}
// O (64 x 128) += P (registers) . V, V MN-major (transposed B)
__device__ __forceinline__ void wg_pv(float (&o)[16][4], const uint32_t (&a)[4], uint32_t blo) {
  asm volatile(
      "{\n .reg .b64 db;\n mov.b64 db, {%68, %69};\n"
      " wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "{%64, %65, %66, %67}, db, 1, 1, 1, 1;\n}\n"
      : "+f"(o[0][0]), "+f"(o[0][1]), "+f"(o[0][2]), "+f"(o[0][3]), "+f"(o[1][0]), "+f"(o[1][1]), "+f"(o[1][2]),
        "+f"(o[1][3]), "+f"(o[2][0]), "+f"(o[2][1]), "+f"(o[2][2]), "+f"(o[2][3]), "+f"(o[3][0]), "+f"(o[3][1]),
        "+f"(o[3][2]), "+f"(o[3][3]), "+f"(o[4][0]), "+f"(o[4][1]), "+f"(o[4][2]), "+f"(o[4][3]), "+f"(o[5][0]),
        "+f"(o[5][1]), "+f"(o[5][2]), "+f"(o[5][3]), "+f"(o[6][0]), "+f"(o[6][1]), "+f"(o[6][2]), "+f"(o[6][3]),
        "+f"(o[7][0]), "+f"(o[7][1]), "+f"(o[7][2]), "+f"(o[7][3]), "+f"(o[8][0]), "+f"(o[8][1]), "+f"(o[8][2]),
        "+f"(o[8][3]), "+f"(o[9][0]), "+f"(o[9][1]), "+f"(o[9][2]), "+f"(o[9][3]), "+f"(o[10][0]), "+f"(o[10][1]),
        "+f"(o[10][2]), "+f"(o[10][3]), "+f"(o[11][0]), "+f"(o[11][1]), "+f"(o[11][2]), "+f"(o[11][3]),
        "+f"(o[12][0]), "+f"(o[12][1]), "+f"(o[12][2]), "+f"(o[12][3]), "+f"(o[13][0]), "+f"(o[13][1]),
        "+f"(o[13][2]), "+f"(o[13][3]), "+f"(o[14][0]), "+f"(o[14][1]), "+f"(o[14][2]), "+f"(o[14][3]),
        "+f"(o[15][0]), "+f"(o[15][1]), "+f"(o[15][2]), "+f"(o[15][3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(blo), "r"(DHI));
}
// swizzled row-major box address: row r (0..63), 16-byte chunk c (0..7) of a 128 B row
__device__ __forceinline__ uint32_t swz(uint32_t base, int r, int c) { return base + r * 128 + ((c ^ (r & 7)) << 4); }

extern "C" __global__ void __launch_bounds__(128, MINB)
top_accum_cuda(const __nv_bfloat16* __restrict__ q, const float* __restrict__ cosc, const float* __restrict__ sinc,
               const int* __restrict__ bitmask, float* __restrict__ out_max, float* __restrict__ out_sum,
               float* __restrict__ out_o, int t_base, int layer_power, int T, int H, int Hkv, int N_layer,
               int NWORDS, float scale, float neg_inf,
               const __grid_constant__ CUtensorMap tmk, const __grid_constant__ CUtensorMap tmv,
               float* __restrict__ gmo, float* __restrict__ gso, float* __restrict__ goo, int FUSE_INIT) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
  const uint32_t sb = smem_u32(smem);
  const uint32_t bar0 = sb + NSTAGE * STAGEB;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int i_blk = blockIdx.x, pid = blockIdx.y;
  const int i_b = pid / Hkv, i_h = pid % Hkv;
  const int tok0 = i_b * N_layer;
  // ---- rows: r0 = lane/4 (query j0 = r0/4, head g = r0%4), r0 + 8 (query j0 + 2) of this warp's 4 queries ----
  const int r0 = lane >> 2, cc = (lane & 3) * 2, j0 = r0 >> 2, hg = r0 & 3;
  const int h = i_h * 4 + hg;
  int tq[2], nc[2];
  bool rok[2];
  long wb[2];
#pragma unroll
  for (int ri = 0; ri < 2; ++ri) {
    int t = i_blk * 16 + 4 * w + j0 + 2 * ri;
    rok[ri] = t < T;
    t = min(t, T - 1);
    tq[ri] = t;
    nc[ri] = (t_base + t) / layer_power + 1;
    wb[ri] = (((long)i_b * T + t) * Hkv + i_h) * NWORDS;
  }
  const int tl_last = min(i_blk * 16 + 15, T - 1);
  const int nc_max = (t_base + tl_last) / layer_power + 1;
  const int n_tiles = (nc_max + TN - 1) / TN;
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) mbar_init(bar0 + 8 * s, 1);
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  auto issue = [&](int it) {           // one thread: the 4 boxes of tile it into stage it % NSTAGE
    const int s = it % NSTAGE;
    const uint32_t st = sb + s * STAGEB, bar = bar0 + 8 * s;
    mbar_expect_tx(bar, STAGEB);
    const int tok = tok0 + it * TN;
    tma3(st, &tmk, 0, i_h, tok, bar);
    tma3(st + 8192, &tmk, 64, i_h, tok, bar);
    tma3(st + 16384, &tmv, 0, i_h, tok, bar);
    tma3(st + 24576, &tmv, 64, i_h, tok, bar);
  };
  if (tid == 0)
    for (int it = 0; it < NSTAGE && it < n_tiles; ++it) issue(it);
  // ---- q rotated to the query's node position, scaled, fp16 A fragments (loop invariant) ----
  uint32_t A[8][4];
  {
    float a[2][8][4];
#pragma unroll
    for (int ri = 0; ri < 2; ++ri) {
      const long pq = (t_base + tq[ri]) / layer_power;
      const __nv_bfloat16* qp = q + (((long)i_b * T + tq[ri]) * H + h) * 128;
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int f = kb * 16 + cc + (i & 1) + (i >> 1) * 8;
          const float ql = __bfloat162float(qp[f]), qh = __bfloat162float(qp[f + 64]);
          const float cq = cosc[pq * 64 + f], sq = sinc[pq * 64 + f];
          a[ri][kb][i] = (ql * cq - qh * sq) * scale;
          a[ri][kb + 4][i] = (ql * sq + qh * cq) * scale;
        }
    }
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      A[kb][0] = pack_h2(a[0][kb][0], a[0][kb][1]);
      A[kb][1] = pack_h2(a[1][kb][0], a[1][kb][1]);
      A[kb][2] = pack_h2(a[0][kb][2], a[0][kb][3]);
      A[kb][3] = pack_h2(a[1][kb][2], a[1][kb][3]);
    }
  }
  float cmax[2] = {neg_inf, neg_inf}, csum[2] = {0.f, 0.f};
  float o[16][4];
#pragma unroll
  for (int nb = 0; nb < 16; ++nb)
#pragma unroll
    for (int i = 0; i < 4; ++i) o[nb][i] = 0.f;

  for (int it = 0; it < n_tiles; ++it) {
    const int s = it % NSTAGE;
    const uint32_t st = sb + s * STAGEB;
    // selected bitmask words of this tile (2 per query), loaded before the wait
    uint32_t wd[2][2];
#pragma unroll
    for (int ri = 0; ri < 2; ++ri)
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int wi = 2 * it + u;
        wd[ri][u] = wi < NWORDS ? (uint32_t)bitmask[wb[ri] + wi] : 0u;
      }
    mbar_wait(bar0 + 8 * s, (it / NSTAGE) & 1);
    // ---- S = q . K^T for 64 nodes: 8 key blocks x 8 dim blocks ----
    float sc[8][4];
#if WG
    {
      const uint32_t lk = dlo(st, 16);
      wg_fence();
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) {
        const uint32_t ko = (kb >> 2) * 512 + (kb & 3) * 2;
        if (kb == 0) wg_qk<0>(sc, A[kb], lk + ko);
        else wg_qk<1>(sc, A[kb], lk + ko);
      }
      wg_commit();
      wg_wait<0>();
    }
#else
#pragma unroll
    for (int nb = 0; nb < 8; ++nb)
#pragma unroll
      for (int i = 0; i < 4; ++i) sc[nb][i] = 0.f;
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      const uint32_t kbase = st + (kb >> 2) * 8192;
      const int ch = (kb & 3) * 2 + ((lane >> 3) & 1);
#pragma unroll
      for (int np = 0; np < 4; ++np) {      // key blocks 2 np, 2 np + 1
        const int key = 16 * np + (lane & 7) + ((lane >> 4) << 3);
        uint32_t b0, b1, b2, b3;
        ldsm_x4(swz(kbase, key, ch), b0, b1, b2, b3);
        mma16816(sc[2 * np], A[kb], b0, b1);
        mma16816(sc[2 * np + 1], A[kb], b2, b3);
      }
    }
#endif
    // ---- mask (causal + not selected), online softmax ----
    float mx[2] = {-INFINITY, -INFINITY};
#if MASK2
    // key kk = nb * 8 + cc + (i & 1): word kk >> 5 = nb >> 2, bit (nb & 3) * 8 + cc + (i & 1) (cc <= 6)
    uint32_t wsh[2][2];
    int lim[2];
#pragma unroll
    for (int ri = 0; ri < 2; ++ri) {
      wsh[ri][0] = wd[ri][0] >> cc;
      wsh[ri][1] = wd[ri][1] >> cc;
      lim[ri] = rok[ri] ? nc[ri] - it * TN - cc : 0;      // node < nc  <=>  nb * 8 + (i & 1) < lim
    }
#pragma unroll
    for (int nb = 0; nb < 8; ++nb)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int ri = i >> 1, off = nb * 8 + (i & 1);
        const bool sel = (wsh[ri][nb >> 2] >> ((nb & 3) * 8 + (i & 1))) & 1u;
        const bool v = off < lim[ri] && !sel;
        sc[nb][i] = v ? sc[nb][i] : -INFINITY;
        mx[ri] = fmaxf(mx[ri], sc[nb][i]);
      }
#else
#pragma unroll
    for (int nb = 0; nb < 8; ++nb)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int ri = i >> 1, kk = nb * 8 + cc + (i & 1), node = it * TN + kk;
        const bool sel = (wd[ri][kk >> 5] >> (kk & 31)) & 1u;
        const bool v = rok[ri] && node < nc[ri] && !sel;
        sc[nb][i] = v ? sc[nb][i] : -INFINITY;
        mx[ri] = fmaxf(mx[ri], sc[nb][i]);
      }
#endif
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
    for (int nb = 0; nb < 8; ++nb)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int ri = i >> 1;
        sc[nb][i] = ex2s((sc[nb][i] - cmax[ri]) * LOG2E);
        ps[ri] += sc[nb][i];
      }
    csum[0] = csum[0] * alpha[0] + ps[0];
    csum[1] = csum[1] * alpha[1] + ps[1];
    if (__any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
      for (int nb = 0; nb < 16; ++nb) {
        o[nb][0] *= alpha[0];
        o[nb][1] *= alpha[0];
        o[nb][2] *= alpha[1];
        o[nb][3] *= alpha[1];
      }
    }
    // ---- O += P . V: 4 key blocks (k16) x 16 dim blocks ----
#if WG
    {
      uint32_t Pf[4][4];
#pragma unroll
      for (int kq = 0; kq < 4; ++kq) {
        Pf[kq][0] = pack_h2(sc[2 * kq][0], sc[2 * kq][1]);
        Pf[kq][1] = pack_h2(sc[2 * kq][2], sc[2 * kq][3]);
        Pf[kq][2] = pack_h2(sc[2 * kq + 1][0], sc[2 * kq + 1][1]);
        Pf[kq][3] = pack_h2(sc[2 * kq + 1][2], sc[2 * kq + 1][3]);
      }
      const uint32_t lv = dlo(st + 16384, 8192);
      wg_fence();
#pragma unroll
      for (int kq = 0; kq < 4; ++kq) wg_pv(o, Pf[kq], lv + kq * 128);
      wg_commit();
      wg_wait<0>();
    }
#else
#pragma unroll
    for (int kq = 0; kq < 4; ++kq) {
      const uint32_t P[4] = {pack_h2(sc[2 * kq][0], sc[2 * kq][1]), pack_h2(sc[2 * kq][2], sc[2 * kq][3]),
                             pack_h2(sc[2 * kq + 1][0], sc[2 * kq + 1][1]), pack_h2(sc[2 * kq + 1][2], sc[2 * kq + 1][3])};
#pragma unroll
      for (int nb = 0; nb < 16; nb += 2) {
        const int key = 16 * kq + (lane & 7) + ((lane >> 3) & 1) * 8, d = nb + (lane >> 4);
        uint32_t b0, b1, b2, b3;
        ldsm_x4_t(swz(st + 16384 + (d >> 3) * 8192, key, d & 7), b0, b1, b2, b3);
        mma16816(o[nb], P, b0, b1);
        mma16816(o[nb + 1], P, b2, b3);
      }
    }
#endif
    __syncthreads();                      // every warp is done with stage s
    if (tid == 0 && it + NSTAGE < n_tiles) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      issue(it + NSTAGE);
    }
  }
#pragma unroll
  for (int ri = 0; ri < 2; ++ri) {
    float sm = csum[ri];
    sm += __shfl_xor_sync(0xffffffffu, sm, 1);
    sm += __shfl_xor_sync(0xffffffffu, sm, 2);
    if (rok[ri]) {
      const long stt = ((long)i_b * T + tq[ri]) * H + h;
      // FUSE_INIT: recsa_merge_to_global_kernel against the fresh accumulators (max neg_inf, sum 0, output 0) keeps
      // the level state when its sum > 1e-10 and the fresh state otherwise
      const bool pc = sm > 1e-10f;
      float* mo = FUSE_INIT ? gmo : out_max;
      float* so = FUSE_INIT ? gso : out_sum;
      float* op = (FUSE_INIT ? goo : out_o) + stt * 128;
      const bool keep = !FUSE_INIT || pc;
      if ((lane & 3) == 0) {
        mo[stt] = keep ? cmax[ri] : neg_inf;
        so[stt] = keep ? sm : 0.f;
      }
#pragma unroll
      for (int nb = 0; nb < 16; ++nb)
        *reinterpret_cast<float2*>(op + nb * 8 + cc) =
            keep ? make_float2(o[nb][2 * ri], o[nb][2 * ri + 1]) : make_float2(0.f, 0.f);
    }
  }
}
