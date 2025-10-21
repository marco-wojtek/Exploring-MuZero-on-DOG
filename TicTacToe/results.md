## TicTacToeV2 results (1000 games against random Bot)

| Network         | Episodes | lr | winrrate | loserate | tierate | Wins as 1 or -1 w 500 games each
|--------------|:-----:|:-------:|:---:|:---:|:---:|:----:|
| SimpleNN |  1k |  0.001   | 69.10% | 27.30% | 3.60% | 400/291
| LargeNN |  1k |  0.001   | 75.60% | 23.60% | 0.80% | 413/343
| ImpNet |  1k |  0.001   | 66.00% | 30.90% | 3.10% | 374/286
| ImpNet |  1.5k |  0.0005   | 79.10% | 20.40% | 0.50% | 453/338
| ImpNet |  2k |  0.0001   | 81.50% | 18.10% | 0.40% | 431/384
| ConvNet |  1k |  0.0001   | 68.50% | 30.60% | 0.90% | 339/346

## TicTacToeV2 results (1000 games against mcts Bot, 5 simulations)

| Network         | Episodes | lr | winrrate | loserate | tierate | Wins as 1 or -1 w 500 games each
|--------------|:-----:|:-------:|:---:|:---:|:---:|:----:|
| SimpleNN |  1k |  0.001   | 3.50% | 94.30% | 2.20% | 13/22
| LargeNN |  1k |  0.001   | 6.70% | 93.30% | 0.00% | 0/67
| ImpNet |  1k |  0.001   | 0.60% | 96.10% | 3.30% | 1/5
| ImpNet |  1.5k |  0.0005   | 1.20% | 98.80% | 0.00% | 0/12
| ImpNet |  2k |  0.0001   | 4.30% | 95.10% | 0.60% | 27/16
| ConvNet |  1k |  0.0001   | 6.60% | 93.20% | 0.20% | 1/65

## TicTacToeV2 results (1000 games against mcts Bot, 10 simulations)

| Network         | Episodes | lr | winrrate | loserate | tierate | Wins as 1 or -1 w 500 games each
|--------------|:-----:|:-------:|:---:|:---:|:---:|:----:|
| SimpleNN |  1k |  0.001   | 2.70% | 95.00% | 2.30% | 13/14
| LargeNN |  1k |  0.001   | 5.70% | 94.30% | 0.00% | 0/57
| ImpNet |  1k |  0.001   | 0.60% | 95.50% | 3.90% | 0/6
| ImpNet |  1.5k |  0.0005   | 3.60% | 96.30% | 0.10% | 17/19
| ImpNet |  2k |  0.0001   | 5.10% | 94.50% | 0.40% | 31/20
| ConvNet |  1k |  0.0001   | 4.50% | 95.30% | 0.20% | 0/45



## TicTacToeV2 results (1000 games against mcts Bot, 30 simulations)

| Network         | Episodes | lr | winrrate | loserate | tierate | Wins as 1 or -1 w 500 games each
|--------------|:-----:|:-------:|:---:|:---:|:---:|:----:|
| SimpleNN |  1k |  0.001   | 2.20% | 96.60% | 1.20% | 15/7
| LargeNN |  1k |  0.001   | 4.40% | 95.60% | 0.00% | 0/44
| ImpNet |  1k |  0.001   | 0.40% | 96.60% | 3.00% | 0/4
| ImpNet |  1.5k |  0.0005   | 0.60% | 99.40% | 0.00% | 0/6
| ImpNet |  2k |  0.0001   | 0.50% | 99.40% | 0.10% | 0/5
| ConvNet |  1k |  0.0001   | 2.10% | 97.80% | 0.10% | 0/21