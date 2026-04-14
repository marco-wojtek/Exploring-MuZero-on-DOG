# FFA agent performances

**With better pin x action assignement**
## seed 31 50 it
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random (12345)|0.05|470/1000|47%|
|random |0.05|460/1000|46%|
|rule-based|0.05|450/1000|45%|
|random MuZero|0.05|551/1000|55.1%|
|versus seed 20 |0.05|295/1000|29.5%|

## seed 31 100 it
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random (12345)|0.05|456/1000|45.6%|
|random |0.05|452/1000|45.2%|
|rule-based|0.05|422/1000|42.2%|
|random MuZero|0.05|674/1000|67.4%|
| versus seed 20 |0.05|250/1000|25%|

## Seed 20
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random (12345)|0.05|403/1000|40.3%|
|random|0.05|402/1000|40.2%|
|rule-based|0.05|361/1000|36.1%|
|random MuZero|0.05|582/1000|58.2%|

## Seed 21
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random (12345)|0.05|85/1000|8.5%|
|random |0.05|106/1000|10.6%|
|rule-based|0.05|93/1000|9.3%|
|random MuZero|0.05|350/1000|35.0%|

---
---
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