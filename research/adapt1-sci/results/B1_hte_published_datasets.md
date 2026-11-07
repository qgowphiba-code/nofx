
### aryl_amination, substrates in the first third of the order (orders [0, 1, 2]): best yield found / best recorded, after n experiments
| method | n=1 | n=5 | n=10 | n=20 | n=30 |
|---|---:|---:|---:|---:|---:|
| random (15 subs) | 0.402 | 0.663 | 0.738 | 0.824 | 0.875 |
| gp_substrate (15 subs) | 0.374 | 0.803 | 0.876 | 0.951 | 0.974 |
| gp_transfer (15 subs) | 0.565 | 0.818 | 0.868 | 0.953 | 0.974 |
| adapt1 (15 subs) | 0.393 | 0.684 | 0.724 | 0.831 | 0.844 |

### aryl_amination, substrates in the remaining two thirds (orders [0, 1, 2]): best yield found / best recorded, after n experiments
| method | n=1 | n=5 | n=10 | n=20 | n=30 |
|---|---:|---:|---:|---:|---:|
| random (30 subs) | 0.436 | 0.747 | 0.820 | 0.879 | 0.912 |
| gp_substrate (30 subs) | 0.484 | 0.841 | 0.911 | 0.952 | 0.976 |
| gp_transfer (30 subs) | 0.852 | 0.937 | 0.974 | 0.988 | 0.994 |
| adapt1 (30 subs) | 0.571 | 0.838 | 0.893 | 0.925 | 0.939 |
paired adapt1 - gp_substrate @n=5 (later substrates): mean -0.003; adapt1 better 17, worse 9, of 30
paired adapt1 - gp_substrate @n=10 (later substrates): mean -0.018; adapt1 better 15, worse 13, of 30
paired adapt1 - gp_transfer @n=5 (later substrates): mean -0.099; adapt1 better 4, worse 22, of 30
paired adapt1 - gp_transfer @n=10 (later substrates): mean -0.081; adapt1 better 1, worse 24, of 30

### suzuki, substrates in the first third of the order (orders [0, 1, 2]): best yield found / best recorded, after n experiments
| method | n=1 | n=5 | n=10 | n=20 | n=30 |
|---|---:|---:|---:|---:|---:|
| random (12 subs) | 0.433 | 0.709 | 0.817 | 0.881 | 0.917 |
| gp_substrate (12 subs) | 0.371 | 0.712 | 0.771 | 0.901 | 0.966 |
| gp_transfer (12 subs) | 0.529 | 0.852 | 0.961 | 0.984 | 0.993 |
| adapt1 (12 subs) | 0.436 | 0.664 | 0.810 | 0.872 | 0.887 |

### suzuki, substrates in the remaining two thirds (orders [0, 1, 2]): best yield found / best recorded, after n experiments
| method | n=1 | n=5 | n=10 | n=20 | n=30 |
|---|---:|---:|---:|---:|---:|
| random (24 subs) | 0.488 | 0.763 | 0.836 | 0.896 | 0.922 |
| gp_substrate (24 subs) | 0.485 | 0.735 | 0.801 | 0.924 | 0.955 |
| gp_transfer (24 subs) | 0.745 | 0.856 | 0.899 | 0.948 | 0.973 |
| adapt1 (24 subs) | 0.484 | 0.758 | 0.915 | 0.947 | 0.959 |
paired adapt1 - gp_substrate @n=5 (later substrates): mean +0.023; adapt1 better 12, worse 11, of 24
paired adapt1 - gp_substrate @n=10 (later substrates): mean +0.114; adapt1 better 15, worse 6, of 24
paired adapt1 - gp_transfer @n=5 (later substrates): mean -0.099; adapt1 better 6, worse 13, of 24
paired adapt1 - gp_transfer @n=10 (later substrates): mean +0.016; adapt1 better 7, worse 10, of 24
