# RecSA: Recursive Sparse Attention

Code for [Recursive Sparse Attention: Million-Token Length Extrapolation at Logarithmic Cost](https://arxiv.org/abs/2611.07232).

RecSA builds a tree over the KV cache. Each node summarizes its subtree with one key and one mean-pooled value; the keys
come from one pooler per layer, shared by all levels and distilled from dense attention. Each level above the tokens
selects at most *k* candidates, so a query costs O(log L) to select and reads O(log L) entries. The summaries of
unselected candidates share one softmax with the exact tokens. Each entry's RoPE position is its index among the
candidates of its level, so every relative distance stays within the range seen in training.

Initialized from Qwen3-4B and trained on 0.62B tokens at 32K, RecSA extrapolates to 2M tokens:

| RULER (13 tasks) | 32K | 131K | 262K | 524K | 1M | 2M |
|---|---|---|---|---|---|---|
| Dense + DCA | 94.6 | 78.5 | 68.4 | 58.0 | 40.0 | 27.6 |
| InfiniteHiP | 79.6 | 63.7 | 57.7 | 49.4 | 36.6 | 34.0 |
| **RecSA** | **94.7** | **83.3** | **75.9** | **69.7** | **60.5** | **49.3** |

At 524K on one H200, RecSA is 3.2x faster than dense attention in prefill and 3.8x faster in decoding.

## Code

The pooler is called `probe` in the code.

| File | Content |
|---|---|
| `recsa/probe_pool.py` | tree construction and the pooler |
| `recsa/parallel.py`, `recsa/backward.py` | prefill kernels (forward and backward): level-by-level selection, attention over summaries and exact tokens |
| `recsa/decode.py`, `recsa/incremental_tree.py` | decoding with an incrementally updated tree and host-memory offload of the leaves |
| `recsa/cuda/` | hand-written CUDA kernels for the selection steps (sm_90) |
| `scripts/graft.py` | RecSA attention in a Hugging Face Qwen3 model |
| `scripts/ruler_eval.py` | RULER evaluation |
| `benchmarks/` | prefill and decoding latency against FlashAttention-2 |

The kernels target H100/H200 (sm_90) with Triton 3.

## Setup

    conda env create -f environment.yml && conda activate recsa
    bash setup/get_model.sh /path/to/qwen3_4b     # Qwen/Qwen3-4B
    export MODEL_DIR=/path/to/qwen3_4b
    bash scripts/run_tests.sh

## Citation

    @article{wang2026recsa,
      title   = {Recursive Sparse Attention: Million-Token Length Extrapolation at Logarithmic Cost},
      author  = {Wang, Ruitian and Yang, Xinyu and Zhang, Minshen and Zhang, Hao},
      journal = {arXiv preprint arXiv:2611.07232},
      year    = {2026}
    }

## License

MIT
