# Seed 30 (now with clear pin action assignment)
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random (12345)|0.0|726/1000|72.6%|
|random|0.0|727/1000|72.7%|
|rule-based|0.0|545/1000|54.5%|
|random MuZero|0.0|891/1000|89.1%|
|Seed 7 (200 it)|0.0|929/1000|92.9%|

Rerun
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random (12345)|0.10|727/1000|72.7%|
|random|0.0|748/1000|74.8%|
|rule-based|0.10|528/1000|52.8%|
|random MuZero|0.10|906/1000|90.6%|
|Seed 7 (200 it)|0.10|79.8/1000|79.8%|

# seed 8
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|357/600|%|
|rule-based|0.0|218/600|%|
|random MuZero|0.0|244/600|%|

# seed 7
## 100 Iterations
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|377/600|62.8%|
|rule-based|0.0|258/600|43%|
|random MuZero|0.0|292/600|48.6%|
## 200 Iterations
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|414/600|69%|
|rule-based|0.0|290/600|48.3%|
|random MuZero|0.0|335/600|55.8%|


# Seed 6
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|347/600|57.8%|
|rule-based|0.0|209/600|34.8%|
|random MuZero|0.0|237/600|39.5%|


# Seed 3
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.2|377/600|56.2%|
|random|0.0|391/600|65.2%|
|rule-based|0.2|252/600|42%|
|rule-based|0.0|265/600|44.2%|
|random MuZero|0.2|276/600|46%|
|random MuZero|0.0|325/600|54.2%|

baseline (rule-based) versus other
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|-|434|72.3%
|random MuZero|1.0|391|65.2%
|random MuZero|0.5|359|59.8%
|random MuZero|0.2|354|59.0%
|random MuZero|0.0|380|63.3%

## Testing Performance versus random in same seed 12345 environment

rule_based = 422/600
random MuZero  = 365/600
seed 3 MuZero = 379/600
seed 6 MuZero = 347/600
seed 7 (100) MuZero = 377/600
seed 7 (200) MuZero = 397/600
seed 8 MuZero = 338/600