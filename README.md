# Parallel Reductions in CUDA

Iteratively optimizing a `reduce_sum` operation in CUDA until we reach >95% of GPU performance. This code accompanies the blog post [Embarrasingly Parallel Reduction in CUDA](https://masterskepticista.github.io/posts/reduce-sum/).

### Results

Effective bandwidth achieved on an RTX-3090 (`N=1<<25` elements):

<div style="width: 70%; margin: auto; align: left">
  <table style="font-size: 0.9em;">
    <thead>
      <tr>
        <th>#</th>
        <th>Kernel</th>
        <th>Bandwidth (GB/s)</th>
        <th>Relative to <code>jnp.sum</code></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>Vector Loads</td>
        <td>9.9</td>
        <td>1.1%</td>
      </tr>
      <tr>
        <td>2</td>
        <td>Interleaved Addressing</td>
        <td>223</td>
        <td>24.7%</td>
      </tr>
      <tr>
        <td>3</td>
        <td>Non-divergent Threads</td>
        <td>317</td>
        <td>36.3%</td>
      </tr>
      <tr>
        <td>4</td>
        <td>Sequential Addressing</td>
        <td>331</td>
        <td>38.0%</td>
      </tr>
      <tr>
        <td>5</td>
        <td>Reduce on First Loads</td>
        <td>618</td>
        <td>70.9%</td>
      </tr>
      <tr>
        <td>6</td>
        <td>Warp Unrolling</td>
        <td>859</td>
        <td>98.6%</td>
      </tr>
      <tr>
        <td>0</td>
        <td><code>jnp.sum</code> reference</td>
        <td>871</td>
        <td>100%</td>
      </tr>
    </tbody>
  </table>
</div>

### Run benchmarks

```bash
# Compile
nvcc -arch=native -O3 --use_fast_math reduce_sum.cu -lcublas -lcublasLt -o ./reduce_sum 

# Run
./reduce_sum <1...6>
```

### Acknowledgements
Benchmarking setup borrowed from [karpathy/llm.c](https://github.com/karpathy/llm.c/).


### License
MIT
