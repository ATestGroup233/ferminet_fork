# from ferminet import base_config
import test_base_config as base_config
from MyConfig import cfg

# from ferminet import constants
import test_constants as constants
import chex
from myLogging import myLogging
import os
from test_optimizer import Optimizer

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

logging = myLogging()

cfg = base_config.resolve(cfg)

import jax
import jax.numpy as jnp

num_devices = jax.device_count()
logging.info(f"Starting QMC with {num_devices} XLA devices.")
data_shape = (num_devices, cfg.batch_size // num_devices)

atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
charges = jnp.array([atom.charge for atom in cfg.system.molecule])
spins = cfg.system.electrons

seed = 23
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)

# Create parameters, network, and vmaped/pmaped derivations
from ferminet import pretrain

# from ferminet import networks
import test_network as networks

hartree_fock = pretrain.get_hf(
    cfg.system.molecule,
    cfg.system.electrons,
    basis=cfg.pretrain.basis,
    restricted=False,
)

hf_solution = None
network_init, signed_network = networks.make_fermi_net(
    atoms,
    spins,
    charges,
    envelope_type=cfg.network.envelope_type,
    bias_orbitals=cfg.network.bias_orbitals,
    use_last_layer=cfg.network.use_last_layer,
    hf_solution=hf_solution,
    full_det=cfg.network.full_det,
    **cfg.network.detnet,
)

from kfac_ferminet_alpha import utils as kfac_utils

params = network_init(subkey)
params = kfac_utils.replicate_all_local_devices(params)


# Often just need log|psi(x)|.
network = lambda params, x: signed_network(params, x)[1]
batch_network = jax.vmap(network, (None, 0), 0)  # batched network

from utils import init_electrons

key, subkey = jax.random.split(key)
data = init_electrons(subkey, cfg.system.molecule, cfg.system.electrons, cfg.batch_size)
data = jnp.reshape(data, data_shape + data.shape[1:])
data = kfac_utils.broadcast_all_local_devices(data)
t_init = 0
opt_state_ckpt = None
mcmc_width_ckpt = None

# Initialisation done. We now want to have different PRNG streams on each
# device. Shard the key over devices
sharded_key = kfac_utils.make_different_rng_key_on_all_devices(key)

# Pretraining to match Hartree-Fock
sharded_key, subkeys = kfac_utils.p_split(sharded_key)
params, data = pretrain.pretrain_hartree_fock(
    params,
    data,
    batch_network,
    subkeys,
    cfg.system.molecule,
    cfg.system.electrons,
    scf_approx=hartree_fock,
    envelope_type=cfg.network.envelope_type,
    full_det=cfg.network.full_det,
    iterations=cfg.pretrain.iterations,
)

# Main training

# Construct MCMC step
from ferminet import mcmc

atoms_to_mcmc = atoms if cfg.mcmc.scale_by_nuclear_distance else None
mcmc_step = mcmc.make_mcmc_step(
    batch_network,
    cfg.batch_size // num_devices,
    steps=cfg.mcmc.steps,
    atoms=atoms_to_mcmc,
    one_electron_moves=cfg.mcmc.one_electron,
)

# Construct loss and optimizer
from utils import make_loss

total_energy = make_loss(
    network, batch_network, atoms, charges, clip_local_energy=cfg.optim.clip_el
)
# Compute the learning rate
def learning_rate_schedule(t):
    return cfg.optim.lr.rate * jnp.power(
        (1.0 / (1.0 + (t / cfg.optim.lr.delay))), cfg.optim.lr.decay
    )


# Differentiate wrt parameters (argument 0)
val_and_grad = jax.value_and_grad(total_energy, argnums=0, has_aux=True)

from kfac_ferminet_alpha import optimizer as kfac_optim

# optimizer = kfac_optim.Optimizer(
optimizer = Optimizer(
    val_and_grad,
    l2_reg=cfg.optim.kfac.l2_reg,
    norm_constraint=cfg.optim.kfac.norm_constraint,
    value_func_has_aux=True,
    learning_rate_schedule=learning_rate_schedule,
    curvature_ema=cfg.optim.kfac.cov_ema_decay,
    inverse_update_period=cfg.optim.kfac.invert_every,
    min_damping=cfg.optim.kfac.min_damping,
    num_burnin_steps=0,
    register_only_generic=cfg.optim.kfac.register_only_generic,
    estimation_mode="fisher_exact",
    multi_device=True,
    pmap_axis_name=constants.PMAP_AXIS_NAME,
    debug=False,
)
sharded_key, subkeys = kfac_utils.p_split(sharded_key)
opt_state = optimizer.init(params, subkeys, data)  #  velocity, estimator, step_counter
opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state

# Only the pmapped MCMC step is needed after this point
mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)

# The actual training loop
import numpy as np

mcmc_width = kfac_utils.replicate_all_local_devices(jnp.asarray(cfg.mcmc.move_width))
pmoves = np.zeros(cfg.mcmc.adapt_frequency)
shared_t = kfac_utils.replicate_all_local_devices(jnp.zeros([]))
shared_mom = kfac_utils.replicate_all_local_devices(jnp.zeros([]))
shared_damping = kfac_utils.replicate_all_local_devices(
    jnp.asarray(cfg.optim.kfac.damping)
)

logging.info(f"Burning in MCMC chain for {cfg.mcmc.burn_in} steps")
for t in range(cfg.mcmc.burn_in):
    sharded_key, subkeys = kfac_utils.p_split(sharded_key)
    data, pmove = mcmc_step(params, data, subkeys, mcmc_width)
logging.info(f"Completed burn-in MCMC steps")
logging.info(f"Initial energy: {constants.pmap(total_energy)(params, data)[0]} E_h")

import time

time_of_last_ckpt = time.time()

for t in range(t_init, cfg.optim.iterations):
    sharded_key, subkeys = kfac_utils.p_split(sharded_key)
    data, pmove = mcmc_step(params, data, subkeys, mcmc_width)
    # Need this split because MCMC step above used subkeys already
    sharded_key, subkeys = kfac_utils.p_split(sharded_key)
    params, opt_state, stats = optimizer.step(  # pytype: disable=attribute-error
        params=params,
        state=opt_state,
        rng=subkeys,
        data_iterator=iter([data]),
        momentum=shared_mom,
        damping=shared_damping,
    )
    loss = stats["loss"]
    aux_data = stats["aux"]

    # due to pmean, loss, variance and pmove should be the same across
    # devices.
    loss = loss[0]
    variance = aux_data.variance[0]
    pmove = pmove[0]

    # Update MCMC move width
    if t > 0 and t % cfg.mcmc.adapt_frequency == 0:
        if np.mean(pmoves) > 0.55:
            mcmc_width *= 1.1
        if np.mean(pmoves) < 0.5:
            mcmc_width /= 1.1
        pmoves[:] = 0
    pmoves[t % cfg.mcmc.adapt_frequency] = pmove

    if cfg.debug.check_nan:
        tree = {"params": params, "loss": loss}
        if cfg.optim.optimizer != "none":
            tree["optim"] = opt_state
        chex.assert_tree_all_finite(tree)

    if t % cfg.log.stats_frequency == 0:
        logging.info(
            "Step %05d: %03.4f E_h, variance=%03.4f E_h^2, pmove=%0.2f"
            % (t, loss, variance, pmove)
        )
