# Parallel Reductions in CUDA

Iteratively optimizing a `reduce_sum` operation in CUDA until we reach >95% of GPU performance. This code accompanies the blog post [Embarrasingly Parallel Reduction in CUDA](https://masterskepticista.github.io/posts/2025/02/reducesum/).

### Results

Effective bandwidth achieved on an RTX-3090 (`N=1<<25` elements):

|   | Kernel                  | Bandwidth (GB/s) | Relative to `jnp.sum` |
|---|-------------------------|------------------|-----------------------|
| 1 | Vector Loads            | 9.9              | 1.1%                  |
| 2 | Interleaved Addressing  | 223              | 24.7%                 |
| 3 | Non-divergent Threads   | 317              | 36.3%                 |
| 4 | Sequential Addressing   | 331              | 38.0%                 |
| 5 | Warp Unrolling          | 550              | 63.4%                 |
| 6 | Batching                | 854              | 98.0%                 |
| 0 | `jnp.sum` reference     | 871              | 100%                  |

### Run benchmarks

```bash
# Compile
nvcc -arch=native -O3 --use_fast_math reduce_sum.cu -lcublas -lcublasLt -o ./reduce_sum 

# Run
./reduce_sum <kernel_num>
```

### Acknowledgements
Benchmarking setup borrowed from [karpathy/llm.c](https://github.com/karpathy/llm.c/).


### License
MIT
