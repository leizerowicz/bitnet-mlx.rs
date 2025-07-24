# MLX vs Candle Performance Comparison Summary

Generated: 2025-07-24 20:02:51 UTC

## matmul

| Tensor Size | Baseline | Comparison | Speedup | Recommendation |
|-------------|----------|------------|---------|----------------|
| 512x512 | candle_cpu | candle_metal | 2902.41x | Use candle for better performance (2902.41x speedup) |
| 512x512 | candle_cpu | candle_metal | 2943.50x | Use candle for better performance (2943.50x speedup) |
| 128x128 | candle_cpu | candle_metal | 185.78x | Use candle for better performance (185.78x speedup) |
| 128x128 | candle_cpu | candle_metal | 168.59x | Use candle for better performance (168.59x speedup) |

## add

| Tensor Size | Baseline | Comparison | Speedup | Recommendation |
|-------------|----------|------------|---------|----------------|
| 128x128 | candle_cpu | candle_metal | 174.75x | Use candle for better performance (174.75x speedup) |
| 128x128 | candle_cpu | candle_metal | 187.08x | Use candle for better performance (187.08x speedup) |
| 512x512 | candle_cpu | candle_metal | 2809.12x | Use candle for better performance (2809.12x speedup) |
| 512x512 | candle_cpu | candle_metal | 3059.01x | Use candle for better performance (3059.01x speedup) |

## Overall Recommendations

- Use candle for better performance (2943.50x speedup): 1 cases
- Use candle for better performance (187.08x speedup): 1 cases
- Use candle for better performance (174.75x speedup): 1 cases
- Use candle for better performance (3059.01x speedup): 1 cases
- Use candle for better performance (185.78x speedup): 1 cases
- Use candle for better performance (2902.41x speedup): 1 cases
- Use candle for better performance (2809.12x speedup): 1 cases
- Use candle for better performance (168.59x speedup): 1 cases
