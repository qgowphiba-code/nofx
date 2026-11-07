
### reactions 0-9: best yield found / grid optimum, after n experiments
| method | n=1 | n=3 | n=5 | n=10 | n=20 |
|---|---:|---:|---:|---:|---:|
| random (10 rx) | 0.355 | 0.616 | 0.751 | 0.830 | 0.927 |
| gp_episode (10 rx) | 0.369 | 0.558 | 0.750 | 0.946 | 0.989 |
| gp_transfer (10 rx) | 0.697 | 0.817 | 0.911 | 0.974 | 0.989 |
| adapt1_noctx (10 rx) | 0.550 | 0.667 | 0.782 | 0.845 | 0.965 |
| adapt1_ctx (10 rx) | 0.344 | 0.588 | 0.773 | 0.848 | 0.923 |

### reactions 10-19: best yield found / grid optimum, after n experiments
| method | n=1 | n=3 | n=5 | n=10 | n=20 |
|---|---:|---:|---:|---:|---:|
| random (10 rx) | 0.438 | 0.634 | 0.691 | 0.847 | 0.940 |
| gp_episode (10 rx) | 0.346 | 0.527 | 0.717 | 0.949 | 0.989 |
| gp_transfer (10 rx) | 0.873 | 0.955 | 0.959 | 1.000 | 1.000 |
| adapt1_noctx (10 rx) | 0.531 | 0.639 | 0.699 | 0.764 | 0.863 |
| adapt1_ctx (10 rx) | 0.531 | 0.833 | 0.968 | 0.996 | 1.000 |

### reactions 20-39: best yield found / grid optimum, after n experiments
| method | n=1 | n=3 | n=5 | n=10 | n=20 |
|---|---:|---:|---:|---:|---:|
| random (20 rx) | 0.430 | 0.646 | 0.742 | 0.863 | 0.942 |
| gp_episode (20 rx) | 0.344 | 0.504 | 0.699 | 0.949 | 0.993 |
| gp_transfer (20 rx) | 0.816 | 0.892 | 0.981 | 0.998 | 1.000 |
| adapt1_noctx (20 rx) | 0.569 | 0.785 | 0.853 | 0.940 | 0.995 |
| adapt1_ctx (20 rx) | 0.688 | 0.905 | 0.933 | 0.985 | 0.995 |
paired adapt1_ctx - gp_episode @n=5, reactions 20-39: mean +0.233, ctx better on 19, worse on 1 of 20
paired adapt1_ctx - gp_episode @n=10, reactions 20-39: mean +0.036, ctx better on 9, worse on 1 of 20
paired adapt1_ctx - gp_transfer @n=5, reactions 20-39: mean -0.049, ctx better on 3, worse on 12 of 20
paired adapt1_ctx - gp_transfer @n=10, reactions 20-39: mean -0.014, ctx better on 1, worse on 4 of 20
paired adapt1_ctx - adapt1_noctx @n=5, reactions 20-39: mean +0.080, ctx better on 11, worse on 2 of 20
paired adapt1_ctx - adapt1_noctx @n=10, reactions 20-39: mean +0.044, ctx better on 11, worse on 1 of 20
