### Experments Conducted in the paper


### Random Search with unknown `C` and `gamma`

- Observations as 10000 as that is the size of our validation set
- According to grid search (Binary)
	- `C` is around  10
	- `gamma` is around 6


SNo.| Observations | Train Error | Test Error | Time Taken (mins) | Search Iters  | Comments
--- | --- | --- | --- | --- | --- | ---
1 | 10000 | 16.309 | 17.42 | 913 | 5000 |{C: 16, g : 5.66} 
2 | 10000 | 16.43| 17.38 | 912 | 5000 | {C: 7.09, g : 7.41} 
3 | 10000 | 16.54| 17.36 | 1805 | 10000 | {C: 5.85, g : 7.58} 
4 | 10000 | 16.43 | 17.44 | 3630 |20000 | {C: 12, g : 5.69} 
5 | 10000 | 16.47 | 17.37 | 5481 | 30000| {C: 6.0588, gamma: 7.704} 
6 | 10000 | 16.54 | 17.5 |  5462 | 30000 | {C: 11.63, gamma: 5.51}
7 | 10000 | 4.99 | 19.66 | 371 | 1000 | {C: 2.423, gamma: 1.884843e-05} **


All experiemnts conducted in Madawaska (Papallel)
RAM : 32 GB
Cores : 8

** Denotes multi class grid search