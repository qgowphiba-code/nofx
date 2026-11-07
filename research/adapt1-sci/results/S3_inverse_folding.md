| method | train ep 0-19 | train ep 20-39 | train ep 40-59 | train ep 60-79 | held-out (frozen) |
|---|---:|---:|---:|---:|---:|
| random (10 seeds) | 0.003 | 0.001 | 0.002 | 0.002 | 0.002 |
| mc_tab (10 seeds) | 0.047 | 0.087 | 0.121 | 0.135 | 0.291 |
| mc_tab_pos (10 seeds) | 0.004 | 0.004 | 0.010 | 0.020 | 0.019 |
| Adapt-1 sequential | 0.001 | 0.000 | 0.001 | 0.029 | 0.156 |
burial heuristic (diagnostic) held-out mean 0.648; exact ceiling held-out mean 0.648; all-target ceiling 0.584
Adapt-1 selected_by (train): [('core_exploration', 960)]
Adapt-1 held-out sequences: [(20, 'HHPHHHHHHHHH', 0.002), (20, 'HHPHHHHHHHHH', 0.002), (21, 'HHPHPHHPHPHH', 0.026), (21, 'HHPHPHHPHPHH', 0.026), (22, 'HHPHPHHHHPHH', 0.008), (22, 'HHPHPHHHHPHH', 0.008), (23, 'HHPHPHHHHPHH', 0.008), (23, 'HHPHPHHHHPHH', 0.008), (24, 'HHPHPHHPHPHH', 0.737), (24, 'HHPHPHHPHPHH', 0.737)]
Adapt-1 fraction H by contact count, last 20 train episodes:
  contacts 0 H fraction 0.45 (n=109)
  contacts 1 H fraction 0.50 (n=66)
  contacts 2 H fraction 0.60 (n=53)
