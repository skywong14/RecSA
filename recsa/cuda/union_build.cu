// Union lists of a block of BQ = 16 queries (TK = 256 selected parents each) in one CTA, sm_90a.
#include <stdint.h>

#define BQ 16
#define TK 256
#define NK (BQ * TK)
#define NT 256                       // threads
#define BIG (1 << 30)
#define PX(x) ((x) + ((x) >> 5))       // padded smem index: per-thread runs of 16 keys without bank conflicts
#define NKP (NK + NK / 32)

extern "C" __global__ void __launch_bounds__(NT) union_build16(
    const int* __restrict__ sel, int* __restrict__ ulist, int* __restrict__ umask, int* __restrict__ ucount,
    int T, int NB, int H_kv, long s_b, long s_t, int U, int PAD, int* __restrict__ posq, int t_base) {
  __shared__ int bufA[NKP];
  __shared__ int bufB[NKP];
  __shared__ int wsum[NT / 32];
  int* key = bufA;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int row = blockIdx.x;                       // (b * NB + blk) * H_kv + h
  const int h = row % H_kv, bb = row / H_kv, blk = bb % NB, b = bb / NB;
  int kv[BQ];
#pragma unroll
  for (int u = 0; u < BQ; ++u) {        // element tid of every query's list (query u)
    const int i = tid + u * NT, t = blk * BQ + u;
    int k = BIG;
    if (t < T) {
      const int v = sel[(long)b * s_b + (long)t * s_t + h * TK + tid];
      if (v >= 0) k = v * BQ + u;
    }
    kv[u] = k;
    key[PX(i)] = k;
  }
  if (posq != nullptr) {
    // merge_acc.query_positions for the leaf (layer_power 1): (valid entries - 1) * CR + global row % CR
#pragma unroll
    for (int u = 0; u < BQ; ++u) {
      const int nv = __syncthreads_count(kv[u] < BIG);
      const int t = blk * BQ + u;
      if (tid == 0 && t < T) posq[((long)b * T + t) * H_kv + h] = (nv - 1) * BQ + (t_base + t) % BQ;
    }
  }
  __syncthreads();
  // each query's run (TK keys) is ascending when its selection is sorted with the empty slots last (the select kernels
  // emit it so): merge the BQ runs pairwise (merge path, 16 outputs per thread); otherwise bitonic-sort everything
  int unsorted = 0;
  for (int i = tid; i < NK; i += NT)
    if ((i % TK) != 0 && key[PX(i - 1)] > key[PX(i)]) unsorted = 1;
  unsorted = __syncthreads_or(unsorted);
  if (!unsorted) {
    int* src = bufA;
    int* dst = bufB;
    for (int L = TK; L < NK; L <<= 1) {
      const int g = tid * (NK / NT);                  // first output of this thread
      const int pr = g / (2 * L), d = g % (2 * L);
      const int a0 = pr * 2 * L, b0 = a0 + L;       // logical starts of the two runs
      int lo = max(0, d - L), hi = min(d, L);         // i = number taken from A
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (src[PX(a0 + mid)] <= src[PX(b0 + d - mid - 1)]) lo = mid + 1; else hi = mid;
      }
      int i = lo, j = d - lo;
      int va = i < L ? src[PX(a0 + i)] : BIG + 1, vb = j < L ? src[PX(b0 + j)] : BIG + 1;
#pragma unroll
      for (int u = 0; u < NK / NT; ++u) {
        const bool takeA = vb > BIG || (va <= BIG && va <= vb);
        dst[PX(g + u)] = takeA ? va : vb;
        if (takeA) { ++i; va = i < L ? src[PX(a0 + i)] : BIG + 1; }
        else { ++j; vb = j < L ? src[PX(b0 + j)] : BIG + 1; }
      }
      __syncthreads();
      int* tmp = src; src = dst; dst = tmp;
    }
    key = src;
  } else {
    for (int k = 2; k <= NK; k <<= 1) {
      for (int j = k >> 1; j > 0; j >>= 1) {
        for (int p = tid; p < NK / 2; p += NT) {
          const int i = 2 * p - (p & (j - 1));        // lower index of the pair with stride j
          const int l = i + j;
          const int a = key[PX(i)], c = key[PX(l)];
          const bool up = (i & k) == 0;
          if ((a > c) == up) {
            key[PX(i)] = c;
            key[PX(l)] = a;
          }
        }
        __syncthreads();
      }
    }
  }
  // segments of equal parent: first flags, masks, exclusive positions
  constexpr int PER = NK / NT;                      // 16 consecutive keys per thread
  const int i0 = tid * PER;
  int nfirst = 0;
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = i0 + u, k = key[PX(i)];
    const int kp = i > 0 ? key[PX(i - 1)] : -BQ;
    nfirst += (k < BIG) && (k / BQ != kp / BQ);
  }
  int incl = nfirst;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, off);
    if (lane >= off) incl += y;
  }
  if (lane == 31) wsum[w] = incl;
  __syncthreads();
  int woff = 0;
  for (int x = 0; x < w; ++x) woff += wsum[x];
  int total = 0;
  for (int x = 0; x < NT / 32; ++x) total += wsum[x];
  int seg = woff + incl - nfirst;
  const long ub = (long)row * U;
#pragma unroll 1
  for (int u = 0; u < PER; ++u) {
    const int i = i0 + u, k = key[PX(i)];
    const int kp = i > 0 ? key[PX(i - 1)] : -BQ;
    if ((k < BIG) && (k / BQ != kp / BQ)) {
      const int par = k / BQ;
      int m = 0;
      for (int x = i; x < NK && key[PX(x)] < BIG && key[PX(x)] / BQ == par; ++x) m |= 1 << (key[PX(x)] % BQ);
      if (seg < U) {
        ulist[ub + seg] = par;
        umask[ub + seg] = m;
      }
      ++seg;
    }
  }
  if (PAD)       // the arrays of union_lists past the count (the CUDA leaf walk reads only the first ucount entries)
    for (int x = total + tid; x < U; x += NT) {
      ulist[ub + x] = BIG;
      umask[ub + x] = 0;
    }
  if (tid == 0) ucount[row] = total;
}

// ---------------------------------------------------------------------------------------------------------------------
// union_build16m: the same arrays as union_build16, built through a parent-indexed mask table in shared memory
// instead of merging the sorted rows. Query u ORs 1 << u into the 16-bit mask of each parent it lists (two parents per
// word: parent p in bits 16 * (p & 1) ..; one row lists distinct parents), then each warp compacts a contiguous run of
// words in parent order (ballots over 32 consecutive words, so the table reads are conflict-free).
// PM = number of parents at this level (> every listed id; the table holds PM / 2 words, dynamic shared memory).
extern "C" __global__ void __launch_bounds__(NT) union_build16m(
    const int* __restrict__ sel, int* __restrict__ ulist, int* __restrict__ umask, int* __restrict__ ucount,
    int T, int NB, int H_kv, long s_b, long s_t, int U, int PAD, int* __restrict__ posq, int t_base, int PM) {
  extern __shared__ __align__(16) unsigned msk[];
  __shared__ int cnt[BQ];
  __shared__ int wred[NT / 32];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int row = blockIdx.x;                       // (b * NB + blk) * H_kv + h
  const int h = row % H_kv, bb = row / H_kv, blk = bb % NB, b = bb / NB;
  int kv[BQ];
  int vmax = -1;
#pragma unroll
  for (int u = 0; u < BQ; ++u) {
    const int t = blk * BQ + u;
    int v = -1;
    if (t < T) v = sel[(long)b * s_b + (long)t * s_t + h * TK + tid];
    kv[u] = v;
    vmax = max(vmax, v);
  }
  vmax = __reduce_max_sync(0xffffffffu, vmax);
  if (tid < BQ) cnt[tid] = 0;
  if (lane == 0) wred[w] = vmax;
  __syncthreads();
  int pm = -1;
#pragma unroll
  for (int x = 0; x < NT / 32; ++x) pm = max(pm, wred[x]);
  const int nw = (min(pm + 1, PM) + 1) >> 1;        // table words in use: parents 0 .. max listed parent
  for (int x = tid; x < (nw + 3) >> 2; x += NT) reinterpret_cast<uint4*>(msk)[x] = make_uint4(0u, 0u, 0u, 0u);
  __syncthreads();
#pragma unroll
  for (int u = 0; u < BQ; ++u) {
    const int v = kv[u];
    if (v >= 0) atomicOr(&msk[v >> 1], (1u << u) << ((v & 1) << 4));
    if (posq != nullptr) {
      const int c = __popc(__ballot_sync(0xffffffffu, v >= 0));
      if (lane == 0 && c) atomicAdd(&cnt[u], c);
    }
  }
  __syncthreads();
  if (posq != nullptr && tid < BQ) {
    // merge_acc.query_positions for the leaf (layer_power 1): (valid entries - 1) * CR + global row % CR
    const int t = blk * BQ + tid;
    if (t < T) posq[((long)b * T + t) * H_kv + h] = (cnt[tid] - 1) * BQ + (t_base + t) % BQ;
  }
  // compaction: warp w owns the words [w0, w1), whole 32-word chunks
  const int R = ((nw + NT / 32 - 1) / (NT / 32) + 31) & ~31, w0 = w * R, w1 = min(w0 + R, nw);
  int n = 0;
  for (int c = w0; c < w1; c += 32) {
    const unsigned m = c + lane < w1 ? msk[c + lane] : 0u;
    n += __popc(__ballot_sync(0xffffffffu, m & 0xffffu)) + __popc(__ballot_sync(0xffffffffu, m >> 16));
  }
  __syncthreads();                                   // wred reuse
  if (lane == 0) wred[w] = n;
  __syncthreads();
  int seg = 0, total = 0;
#pragma unroll
  for (int x = 0; x < NT / 32; ++x) {
    seg += x < w ? wred[x] : 0;
    total += wred[x];
  }
  const long ub = (long)row * U;
  const unsigned lt = (1u << lane) - 1u;
  for (int c = w0; c < w1; c += 32) {
    const unsigned m = c + lane < w1 ? msk[c + lane] : 0u;
    const unsigned lo = m & 0xffffu, hi = m >> 16;
    const unsigned blo = __ballot_sync(0xffffffffu, lo), bhi = __ballot_sync(0xffffffffu, hi);
    int s = seg + __popc(blo & lt) + __popc(bhi & lt);
    if (lo) {
      if (s < U) {
        ulist[ub + s] = 2 * (c + lane);
        umask[ub + s] = (int)lo;
      }
      ++s;
    }
    if (hi && s < U) {
      ulist[ub + s] = 2 * (c + lane) + 1;
      umask[ub + s] = (int)hi;
    }
    seg += __popc(blo) + __popc(bhi);
  }
  if (PAD)
    for (int x = total + tid; x < U; x += NT) {
      ulist[ub + x] = BIG;
      umask[ub + x] = 0;
    }
  if (tid == 0) ucount[row] = total;
}

// ---------------------------------------------------------------------------------------------------------------------
// union_xmask16: merge_acc.union_xmask for BQ = 16, TK = 256: xmask[row, j * 16 + qi] =
// OR of 1 << (n % CR) over the nodes n >= 0 query qi of the block selected, j = lower_bound(row's ulist[0:cnt], n / CR)
// (dropped when j == cnt). The lower bounds come from a parent-indexed table in shared memory (parents < PM; larger
// ids by binary search), the masks are ORed in shared memory and the whole row is written (no zero fill). CR = 1 << CRS.
extern "C" __global__ void __launch_bounds__(NT) union_xmask16(
    const int* __restrict__ nsel, const int* __restrict__ ulist, const int* __restrict__ ucount, int* __restrict__ xmask,
    int T, int NB, int H_kv, long s_b, long s_t, int U, int PM, int CRS) {
  extern __shared__ __align__(16) int xsm[];
  int* xs = xsm;                                              // [U * BQ] masks
  unsigned short* jt = reinterpret_cast<unsigned short*>(xsm + U * BQ);   // [PM] lower bounds
  const int tid = threadIdx.x;
  const int row = blockIdx.x;                                 // (b * NB + blk) * H_kv + h
  const int h = row % H_kv, bb = row / H_kv, blk = bb % NB, b = bb / NB;
  const int cnt = ucount[row];
  const int* lst = ulist + (long)row * U;
  for (int x = tid; x < U * BQ; x += NT) xs[x] = 0;
  for (int p = tid; p < PM; p += NT) jt[p] = (unsigned short)cnt;
  __syncthreads();
  for (int j = tid; j < cnt; j += NT) {                       // parents (list[j - 1], list[j]] -> j
    const int lo = j ? lst[j - 1] + 1 : 0, hi = min(lst[j], PM - 1);
    for (int p = lo; p <= hi; ++p) jt[p] = (unsigned short)j;
  }
  __syncthreads();
  const int cmask = (1 << CRS) - 1;
#pragma unroll 4
  for (int qi = 0; qi < BQ; ++qi) {
    const int t = blk * BQ + qi;
    if (t >= T) break;
    const int n = nsel[(long)b * s_b + (long)t * s_t + h * TK + tid];
    if (n >= 0) {
      const int p = n >> CRS;
      int j;
      if (p < PM) {
        j = jt[p];
      } else {                                                // lower bound by bisection (as the Triton kernel)
        int lo = 0, hi = cnt;
        while (lo < hi) {
          const int mid = (lo + hi) >> 1;
          if (lst[mid] < p) lo = mid + 1;
          else hi = mid;
        }
        j = lo;
      }
      if (j < cnt) atomicOr(&xs[j * BQ + qi], 1 << (n & cmask));
    }
  }
  __syncthreads();
  int4* xo = reinterpret_cast<int4*>(xmask + (long)row * U * BQ);
  const int4* xi = reinterpret_cast<const int4*>(xs);
  for (int x = tid; x < U * BQ / 4; x += NT) xo[x] = xi[x];
}
