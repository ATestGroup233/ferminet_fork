# ferminet notes

运行一下测试用例的配置文件：

```bash
python ferminet.py --config ferminet/configs/atom.py --config.system.atom Li --config.batch_size 256 --config.pretrain.iterations 100
```

```bash
[fdd] output config: 
batch_size: 256
config_module: .atom
debug:
  check_nan: false
  deterministic: false
log:
  features: false
  local_energies: false
  restore_path: ''
  save_frequency: 10.0
  save_path: ''
  stats_frequency: 1
  walkers: false
mcmc:
  adapt_frequency: 100
  burn_in: 100
  init_means: !!python/tuple []
  init_width: 0.8
  move_width: 0.02
  num_leapfrog_steps: 10
  one_electron: false
  scale_by_nuclear_distance: false
  steps: 10
  use_hmc: false
network:
  bias_orbitals: false
  detnet:
    after_determinants: !!python/tuple
    - 1
    determinants: 16
    hidden_dims: !!python/tuple
    - &id001 !!python/tuple
      - 256
      - 32
    - *id001
    - *id001
    - *id001
  envelope_type: full
  full_det: true
  use_last_layer: false
optim:
  adam:
    b1: 0.9
    b2: 0.999
    eps: 1.0e-08
    eps_root: 0.0
  clip_el: 5.0
  iterations: 1000000
  kfac:
    cov_ema_decay: 0.95
    cov_update_every: 1
    damping: 0.001
    invert_every: 1
    l2_reg: 0.0
    mean_center: true
    min_damping: 0.0001
    momentum: 0.0
    momentum_type: regular
    norm_constraint: 0.001
    register_only_generic: false
  lr:
    decay: 1.0
    delay: 10000.0
    rate: 0.0001
  optimizer: kfac
pretrain:
  basis: sto-6g
  iterations: 100
  method: hf
system:
  atom: Li
  charge: 0
  delta_charge: 0.0
  electrons: !!python/tuple
  - 2
  - 1
  molecule:
  - !!python/object:ferminet.utils.system.Atom
    atomic_number: 3
    charge: 3.0
    coords: !!python/tuple
    - 0.0
    - 0.0
    - 0.0
    symbol: Li
    units: bohr
  ndim: 3
  pyscf_mol: null
  set_molecule: _adjust_nuclear_charge
  spin_polarisation: null
  type: 0
  units: bohr

[fdd] finish output
```

可以看出，ferminet的配置分为如下几个方面：

+ mcmc： 蒙卡
+ network： 网络部分
+ optim： 优化器
+ system：待计算原子分子描述
