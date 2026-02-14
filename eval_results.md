## 6115 vs 6115
600 games
 [[118  32 118  32]
 [ 34 116  34 116]
 [127  23 127  23]
 [ 31 119  31 119]]
-> Team that starts wins > 60 \%
# Comparison of all TEAM agents up to 11.02.2026
## Wins versus random init MuZero (except 6001)
| seed | 6115 | 6003 | 6005 | 
|---------:|:--------:|:--------:|:--------:|
| wins (800 total) | 624 | 514 | 350 |
|---------:|:--------:|:--------:|:--------:|
| wins (%) | 78% | 64.24% | 43.75% |

## All agents versus 6115
| seed | 6001 | 6003 | 6005 | 
|---------:|:--------:|:--------:|:--------:|
| wins (800 total) | 257 | 325 | 118 |
|---------:|:--------:|:--------:|:--------:|
| wins (%) | 32.125% | 40.625% | 14.75% |

6003 vs 6001 : 272 - 528


# Evaluation of 6001 and 6115 vs untrained and each other (10.02.2026)
## 6115 vs untrained (P2&4)
### 6115 trained similar to 6001 but for 160 instead of 100 iterations
Final Results

Total Wins per Player and different Starters

| Starter | Player 0 | Player 1 | Player 2 | Player 3 |
|---------:|:--------:|:--------:|:--------:|:--------:|
| Starter 0 | 21 | 129 | 21 | 129 |
| Starter 1 | 25 | 125 | 25 | 125 |
| Starter 2 | 30 | 120 | 30 | 120 |
| Starter 3 | 25 | 125 | 25 | 125 |

Total Wins per Player

| Player | Wins |
|:-------|-----:|
| Player 0 | 101 |
| Player 1 | 499 |
| Player 2 | 101 |
| Player 3 | 499 |

Average Final Pin distance per Player and different Starters

| Starter | Player 0 | Player 1 | Player 2 | Player 3 |
|:--------|:--------:|:--------:|:--------:|:--------:|
| Starter 0 | 55.633335 | 3.3600001 | 53.373333 | 4.1933336 |
| Starter 1 | 55.606667 | 3.68      | 54.54     | 3.8000002 |
| Starter 2 | 54.666668 | 5.386667  | 47.7      | 3.5733335 |
| Starter 3 | 47.90667  | 5.94      | 49.2      | 5.1333337 |

Average Final Pin distance per Player

| Player | Avg Distance |
|:-------|-------------:|
| Player 0 | 53.45334 |
| Player 1 | 4.5916667 |
| Player 2 | 51.20333 |
| Player 3 | 4.175 |
## 6001 vs untrained (P2&4)
Final Results

Total Wins per Player and different Starters

| Starter | Player 0 | Player 1 | Player 2 | Player 3 |
|---------:|:--------:|:--------:|:--------:|:--------:|
| Starter 0 | 48 | 102 | 48 | 102 |
| Starter 1 | 35 | 115 | 35 | 115 |
| Starter 2 | 40 | 110 | 40 | 110 |
| Starter 3 | 44 | 106 | 44 | 106 |

Total Wins per Player

| Player | Wins |
|:-------|-----:|
| Player 0 | 167 |
| Player 1 | 433 |
| Player 2 | 167 |
| Player 3 | 433 |

Average Final Pin distance per Player and different Starters

| Starter | Player 0 | Player 1 | Player 2 | Player 3 |
|:--------|:--------:|:--------:|:--------:|:--------:|
| Starter 0 | 36.033333 | 9.886667 | 39.58 | 7.3933334 |
| Starter 1 | 36.713333 | 5.96     | 38.446667 | 5.82 |
| Starter 2 | 47.600002 | 9.246667 | 47.253334 | 9.16 |
| Starter 3 | 37.46     | 7.8133335| 40.233334 | 8.913334 |

Average Final Pin distance per Player

| Player | Avg Distance |
|:-------|-------------:|
| Player 0 | 39.451668 |
| Player 1 | 8.226667 |
| Player 2 | 41.378334 |
| Player 3 | 7.8216667 |
## 6001 (P2&4) vs 6115 (P1&3)
Final Results

Total Wins per Player and different Starters

| Starter | Player 0 | Player 1 | Player 2 | Player 3 |
|---------:|:--------:|:--------:|:--------:|:--------:|
| Starter 0 | 103 | 47 | 103 | 47 |
| Starter 1 | 86  | 64 | 86  | 64 |
| Starter 2 | 104 | 46 | 104 | 46 |
| Starter 3 | 98  | 52 | 98  | 52 |

Total Wins per Player

| Player | Wins |
|:-------|-----:|
| Player 0 | 391 |
| Player 1 | 209 |
| Player 2 | 391 |
| Player 3 | 209 |

Average Final Pin distance per Player and different Starters

| Starter | Player 0 | Player 1 | Player 2 | Player 3 |
|:--------|:--------:|:--------:|:--------:|:--------:|
| Starter 0 | 5.786667  | 27.566668 | 8.820001  | 30.826668 |
| Starter 1 | 12.626667 | 20.213333 | 12.206667 | 26.060001 |
| Starter 2 | 10.146667 | 31.960001 | 9.8       | 29.873335 |
| Starter 3 | 11.993334 | 23.946667 | 9.900001  | 23.6      |

Average Final Pin distance per Player

| Player | Avg Distance |
|:-------|-------------:|
| Player 0 | 10.138334 |
| Player 1 | 25.921665 |
| Player 2 | 10.181667 |
| Player 3 | 27.59 |

Evaluation completed in 584.62 seconds.



# Evaluation of 6001 versus untrained MuZero nets
## Evaluation 6001 - Run 0

### Total Wins per Player and different Starters

| Starter | Player 0 Wins | Player 1 Wins | Player 2 Wins | Player 3 Wins |
|---------|---------------|---------------|---------------|---------------|
| Player 0 | 72 | 28 | 72 | 28 |
| Player 1 | 64 | 36 | 64 | 36 |
| Player 2 | 51 | 49 | 51 | 49 |
| Player 3 | 65 | 35 | 65 | 35 |

**Total Wins per Player:** Player 0: 252, Player 1: 148, Player 2: 252, Player 3: 148

**Total Wins per Player (second iter):** Player 0: 243, Player 1: 157, Player 2: 243, Player 3: 157

### Average Final Pin Distance per Player and different Starters

| Starter | Player 0 | Player 1 | Player 2 | Player 3 |
|---------|----------|----------|----------|----------|
| Player 0 | 12.28 | 37.23 | 9.64 | 38.47 |
| Player 1 | 11.22 | 37.22 | 15.02 | 36.65 |
| Player 2 | 16.59 | 21.18 | 21.02 | 25.13 |
| Player 3 | 12.33 | 27.07 | 13.81 | 35.67 |

**Average Final Pin Distance per Player:** Player 0: 13.11, Player 1: 30.68, Player 2: 14.87, Player 3: 33.98

---

## Evaluation 6001 - Run 1
### Total Wins per Player and different Starters

| Starter | Player 0 Wins | Player 1 Wins | Player 2 Wins | Player 3 Wins |
|---------|---------------|---------------|---------------|---------------|
| Player 0 | 59 | 41 | 59 | 41 |
| Player 1 | 59 | 41 | 59 | 41 |
| Player 2 | 63 | 37 | 63 | 37 |
| Player 3 | 62 | 38 | 62 | 38 |

**Total Wins per Player:** Player 0: 243, Player 1: 157, Player 2: 243, Player 3: 157

### Average Final Pin Distance per Player and different Starters

| Starter | Player 0 | Player 1 | Player 2 | Player 3 |
|---------|----------|----------|----------|----------|
| Player 0 | 15.49 | 27.29 | 10.45 | 25.88 |
| Player 1 | 18.30 | 24.35 | 17.83 | 24.45 |
| Player 2 | 13.92 | 30.19 | 17.75 | 32.86 |
| Player 3 | 14.06 | 30.10 | 16.39 | 32.62 |

**Average Final Pin Distance per Player:** Player 0: 15.44, Player 1: 27.98, Player 2: 15.61, Player 3: 28.95

---

## Evaluation 6001 - Run 2 (Only P2 trained)

### Total Wins per Player and different Starters

| Starter | Player 0 Wins | Player 1 Wins | Player 2 Wins | Player 3 Wins |
|---------|---------------|---------------|---------------|---------------|
| Player 0 | 55 | 45 | 55 | 45 |
| Player 1 | 61 | 39 | 61 | 39 |
| Player 2 | 54 | 46 | 54 | 46 |
| Player 3 | 54 | 46 | 54 | 46 |

**Total Wins per Player:** Player 0: 224, Player 1: 176, Player 2: 224, Player 3: 176

### Average Final Pin Distance per Player and different Starters

| Starter | Player 0 | Player 1 | Player 2 | Player 3 |
|---------|----------|----------|----------|----------|
| Player 0 | 25.78 | 29.86 | 24.25 | 25.02 |
| Player 1 | 19.78 | 27.92 | 16.59 | 33.81 |
| Player 2 | 27.03 | 32.64 | 23.57 | 26.92 |
| Player 3 | 25.14 | 24.37 | 20.54 | 27.51 |

**Average Final Pin Distance per Player:** Player 0: 24.43, Player 1: 28.70, Player 2: 21.24, Player 3: 28.31
---
# Evaluation of 6003 versus untrained MuZero nets
## Evaluation 6003 - Run 0

**Model trained with larger max eps**

### Total Wins per Player and different Starters

| Starter | Player 0 Wins | Player 1 Wins | Player 2 Wins | Player 3 Wins |
|---------|---------------|---------------|---------------|---------------|
| Player 0 | 46 | 54 | 46 | 54 |
| Player 1 | 48 | 52 | 48 | 52 |
| Player 2 | 50 | 50 | 50 | 50 |
| Player 3 | 56 | 44 | 56 | 44 |

**Total Wins per Player:** Player 0: 200, Player 1: 200, Player 2: 200, Player 3: 200

### Average Final Pin Distance per Player and different Starters

| Starter | Player 0 | Player 1 | Player 2 | Player 3 |
|---------|----------|----------|----------|----------|
| Player 0 | 24.17 | 19.38 | 26.69 | 18.27 |
| Player 1 | 22.47 | 20.62 | 22.16 | 28.31 |
| Player 2 | 24.18 | 26.60 | 25.66 | 29.72 |
| Player 3 | 19.55 | 23.02 | 16.96 | 23.73 |

**Average Final Pin Distance per Player:** Player 0: 22.59, Player 1: 22.40, Player 2: 22.87, Player 3: 25.01

---

## Evaluation 6003 - Run 1

### Total Wins per Player and different Starters

| Starter | Player 0 Wins | Player 1 Wins | Player 2 Wins | Player 3 Wins |
|---------|---------------|---------------|---------------|---------------|
| Player 0 | 46 | 54 | 46 | 54 |
| Player 1 | 56 | 44 | 56 | 44 |
| Player 2 | 54 | 46 | 54 | 46 |
| Player 3 | 59 | 41 | 59 | 41 |

**Total Wins per Player:** Player 0: 215, Player 1: 185, Player 2: 215, Player 3: 185

### Average Final Pin Distance per Player and different Starters

| Starter | Player 0 | Player 1 | Player 2 | Player 3 |
|---------|----------|----------|----------|----------|
| Player 0 | 22.53 | 18.04 | 26.38 | 18.36 |
| Player 1 | 21.98 | 26.41 | 19.03 | 23.08 |
| Player 2 | 23.76 | 22.87 | 22.09 | 24.22 |
| Player 3 | 17.91 | 23.69 | 24.14 | 21.64 |

**Average Final Pin Distance per Player:** Player 0: 21.54, Player 1: 22.75, Player 2: 22.91, Player 3: 21.82

---

## Evaluation 6003 - Run 2 (Only P2 trained)

### Total Wins per Player and different Starters

| Starter | Player 0 Wins | Player 1 Wins | Player 2 Wins | Player 3 Wins |
|---------|---------------|---------------|---------------|---------------|
| Player 0 | 47 | 53 | 47 | 53 |
| Player 1 | 46 | 54 | 46 | 54 |
| Player 2 | 50 | 50 | 50 | 50 |
| Player 3 | 48 | 52 | 48 | 52 |

**Total Wins per Player:** Player 0: 191, Player 1: 209, Player 2: 191, Player 3: 209

### Average Final Pin Distance per Player and different Starters

| Starter | Player 0 | Player 1 | Player 2 | Player 3 |
|---------|----------|----------|----------|----------|
| Player 0 | 29.39 | 25.50 | 30.12 | 23.35 |
| Player 1 | 27.67 | 24.29 | 31.61 | 28.77 |
| Player 2 | 27.76 | 27.12 | 23.17 | 28.32 |
| Player 3 | 33.08 | 25.88 | 35.10 | 25.16 |

**Average Final Pin Distance per Player:** Player 0: 29.47, Player 1: 25.70, Player 2: 30.00, Player 3: 26.40

---

**Evaluation completed in 5716.93 seconds.**


## Game Length Random Rounds
### No circular Boards with Teams

**Stochastic**:

Max game length: 790
Games longer than 600 steps: 9
Evaluation completed in 298.71 seconds.

**Deterministic**:

Average game length: 395.4925
Max game length: 658
Games longer than 600 steps: 3
### Circular Boards with Teams

**Stochastic**:
Average game length: 612.515
Max game length: 1124
Games longer than 600 steps: 198

**Deterministic**:
Average game length: 539.75
Max game length: 1087
Games longer than 600 steps: 177

### Circular Boards without Teams
**Stochastic**:
Average game length: 493.1325
Max game length: 1032
Games longer than 600 steps: 91

**Deterministic**:
Average game length: 516.61
Max game length: 963
Games longer than 600 steps: 97