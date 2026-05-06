| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0||%|
|None|0.0||%|

## Seed 28 (Classic Stochastic MuZero with non-circular trained and belief states)
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|904/1000|90.4%|
|None|0.0|925/1000|92.5%|
|28, 50 its|550/1000|55%|

# Session based Agents
## Seed 25 (belief states + Bootstrapping)
**(C) := Circular board in evaluation**
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|577/600|96.1%|
|None|0.0|583/600|97.1%|
|(C) random|0.0|758/800|94.7%|
|(C) None|0.0|388/400|97%|
|(C) seed 25 it 50 |0.0|238/400|59.5%|
|(C) seed 25 it 150 (ft on C)|0.0|327/400|81.7%|

## Seed 27 (belief states + no bootstrapping)
**(C) := Circular board in evaluation**
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|952/1000|95.2%|
|None|0.0|943/1000|94.3%|
|(C) random|0.0|375/400|93.75%|

## seed 25 vs 27
| Setup | Temp | Wins for 25 | % |
|---------:|-----:|-----:|--:|
|Circular enabled|0.0|224/400|56%|
|Circular disabled|0.0|189/400|47.2%|


## Seed 24 100 it (trained with circular disabled and evaluated with circular disabled)
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|558/600|93%|
|None|0.0|578/600|96.3%|
|seed 23|0.0|402/600|67%|
**With circular enabled**
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|271/300|90.3%|
|None|0.0|281/300|93.6%|
|seed 23|0.0|206/300|68.6%| 

## Seed 23 100 it (trained with circular disabled and evaluated with circular enabled)
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|239/300|79.6%|
|None|0.0|245/300|81.6%|
|it 50|0.0|176/300|58.6%|

**With circular rule disabled in evaluation**
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|260/300|86.6%|
|None|0.0|282/300|94%|
|it 50|0.0|177/300|59%|


# Static Chance Agents
## seed 210 (300 it in total) (trained with circular disabled and evaluated with circular enabled)
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|153/200|76.5%|
|None|0.0|158/200|79%|
|it 100|0.0|123/200|61.5%|

## seed 210 (200 it in total) (trained with circular disabled and evaluated with circular enabled)
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|143/200|71.5%|
|None|0.0|136/200|68%|

## seed 210 (100 ot in total) (trained with circular disabled and evaluated with circular enabled)
| Opponent | Temp | Wins | % | 
|---------:|-----:|-----:|--:|
|random|0.0|123/200|61.5%|
|None|0.0|135/200|67.5%|

## seed 21 (trained with circular disabled and evaluated with circular enabled)
| Opponent | Temp | Wins | % |
|---------:|-----:|-----:|--:|
|random|0.0|38/200|19%|