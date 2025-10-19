## TicTacToeV2 results (1000 games against random Bot)

| Network         | Episodes | lr | winrrate | loserate | tierate | Wins as 1 or -1 w 500 games each
|--------------|:-----:|-------:|:---:|:---:|:---:|:----:|
| SimpleNN |  1k |  0.001   | 69.10% | 27.30% | 3.60% | 400/291
| LargeNN |  1k |  0.001   | 75.60% | 23.60% | 0.80% | 413/343
| ImpNet |  1k |  0.001   | 66.00% | 30.90% | 3.10% | 374/286
| ImpNet |  1.5k |  0.0005   | 79.10% | 20.40% | 0.50% | 453/338
| ImpNet |  2k |  0.0001   | 81.50% | 18.10% | 0.40% | 431/384
| ConvNet |  1k |  0.0001   | 68.50% | 30.60% | 0.90% | 339/346
