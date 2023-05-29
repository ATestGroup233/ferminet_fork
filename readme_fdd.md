# ferminet

采用深度学习方法求解多体薛定谔方程

+ useful links
  + [github](https://github.com/deepmind/ferminet)
  + [paper tensorflow](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033429)
  + [paper jax](https://arxiv.org/abs/2011.07125)

## 安装

1. 下载仓库

    ```bash
    git clone https://github.com/ATestGroup233/ferminet_fork
    ```

2. 本地安装

    ```bash
    git checkout jax
    pip install -e . 
    ```

3. 运行测试

    ```bash
    python ferminet --config H2_config.py
    ```

## 变量解释

1. data(num_device, batch_size, coords): 最后一个维度表示的是系统每个电子的三个坐标，比如有三个电子，最后一个维度是9
2. MCMC指的是Markov Chain Monte Carlo，一种采样方法，这里应该是指采样电子的坐标。相对于蒙卡sampling，这种方法在高维空间具有更好的性质
    + [link](https://machinelearningmastery.com/markov-chain-monte-carlo-for-probability/)
    + `burn in`: 是MCMC里面的术语，因为MCMC对初始值很敏感，因此需要一个`warm-up`或者`burn-in`的过程来将采样空间移动到`正确的`的搜索空间
    + Gibbs Algorithm：下一次采样的概率为上一次采样概率的条件概率
    + Metropolis-Hastings Algorithm: 在条件概率无法计算的情况下（仅适用于Gibbs采样），都可以使用MHA采样。MHA采样思想是通过一个阈值来判断新采样的样本保留或者丢弃，这个阈值是一个概率，它根据采样分布于真实分布的差异来计算。
3. params: ferminet中的网络参数，字典类型，`keys = {'single', 'double', 'orbital', 'envelope'}`

## 研读ferminet笔记

1. ferminet的网络一共有13个blocks，按照从前往后的顺序分别为9个RepeatedDenseBlock+QmCBlockedDens_0+scaleAndShiftDiagonal_0+QmCBlockedDens_1+scaleAndShiftDiagonal_1；因此其参数也分别对应13个blocks
2. ferminet网络的输入为(batch_size, 3*num_electrons)，注意batch_size维度会被平分到各个TPU/GPU上

## QA

1. 时不时会出现以下报错信息，将train.py的314行`key, subkey = jax.random.split(key)`挪到291行`key = jax.random.PRNGKey(seed)`后面.(PS:不要问我为什么，我也一脸懵逼)

    ```bash
    Fatal Python error: Segmentation fault

    Thread 0x00007f6a68edb740 (most recent call first):
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/jax/interpreters/xla.py", line 352 in backend_compile
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/jax/interpreters/xla.py", line 727 in _xla_callable
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/jax/linear_util.py", line 260 in memoized_fun
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/jax/interpreters/xla.py", line 580 in _xla_call_impl
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/jax/core.py", line 631 in process_call
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/jax/core.py", line 1278 in process
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/jax/core.py", line 1266 in call_bind
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/jax/core.py", line 1275 in bind
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/jax/api.py", line 289 in cache_miss
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/jax/api.py", line 398 in f_jitted
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/jax/_src/traceback_util.py", line 139 in reraise_with_filtered_traceback
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/jax/_src/random.py", line 259 in split
    File "/home/fdd/projects/ferminet_fork/ferminet/train.py", line 314 in train
    File "/home/fdd/projects/ferminet_fork/ferminet.py", line 35 in main
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/absl/app.py", line 251 in _run_main
    File "/home/fdd/anaconda3/envs/jax_cuda112/lib/python3.9/site-packages/absl/app.py", line 303 in run
    File "/home/fdd/projects/ferminet_fork/ferminet.py", line 42 in <module>
    Segmentation fault (core dumped)
    ```
