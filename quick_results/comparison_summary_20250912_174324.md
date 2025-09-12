# MLX vs Candle Performance Comparison Summary

Generated: 2025-09-12 17:43:24 UTC

## matmul

| Tensor Size | Baseline | Comparison | Speedup | Recommendation |
|-------------|----------|------------|---------|----------------|
| 128x128 | candle_cpu | candle_metal | 156.82x | Use candle for better performance (156.82x speedup) |
| 128x128 | candle_cpu | candle_metal | 173.37x | Use candle for better performance (173.37x speedup) |
| 512x512 | candle_cpu | candle_metal | 2527.51x | Use candle for better performance (2527.51x speedup) |
| 512x512 | candle_cpu | candle_metal | 2465.97x | Use candle for better performance (2465.97x speedup) |

## add

| Tensor Size | Baseline | Comparison | Speedup | Recommendation |
|-------------|----------|------------|---------|----------------|
| 128x128 | candle_cpu | candle_metal | 163.24x | Use candle for better performance (163.24x speedup) |
| 128x128 | candle_cpu | candle_metal | 165.90x | Use candle for better performance (165.90x speedup) |
| 512x512 | candle_cpu | candle_metal | 2576.64x | Use candle for better performance (2576.64x speedup) |
| 512x512 | candle_cpu | candle_metal | 2522.15x | Use candle for better performance (2522.15x speedup) |

## Overall Recommendations

- Use candle for better performance (163.24x speedup): 1 cases
- Use candle for better performance (2527.51x speedup): 1 cases
- Use candle for better performance (165.90x speedup): 1 cases
- Use candle for better performance (2576.64x speedup): 1 cases
- Use candle for better performance (173.37x speedup): 1 cases
- Use candle for better performance (2522.15x speedup): 1 cases
- Use candle for better performance (156.82x speedup): 1 cases
- Use candle for better performance (2465.97x speedup): 1 cases
