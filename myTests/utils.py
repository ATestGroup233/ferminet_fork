import jax 
import jax.numpy as jnp 
from typing import Sequence
import numpy as np 
from ferminet.utils import system

from ferminet import constants
from ferminet import hamiltonian
import chex

@chex.dataclass
class AuxiliaryLossData:
  """Auxiliary data returned by total_energy.

  Attributes:
    variance: mean variance over batch, and over all devices if inside a pmap.
    local_energy: local energy for each MCMC configuration.
  """
  variance: jnp.DeviceArray
  local_energy: jnp.DeviceArray



def init_electrons(
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
) -> jnp.ndarray:
  """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    molecule: system.Atom objects making up the molecule.
    electrons: tuple of number of alpha and beta electrons.
    batch_size: total number of MCMC configurations to generate across all
      devices.

  Returns:
    array of (batch_size, nalpha*nbeta*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3).
  """
  if sum(atom.charge for atom in molecule) != sum(electrons):
    if len(molecule) == 1:
      atomic_spin_configs = [electrons]
    else:
      raise NotImplementedError('No initialization policy yet '
                                'exists for charged molecules.')
  else:
    atomic_spin_configs = [
        (atom.element.nalpha, atom.element.nbeta) for atom in molecule
    ]
    assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
    while tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons:
      i = np.random.randint(len(atomic_spin_configs))
      nalpha, nbeta = atomic_spin_configs[i]
      atomic_spin_configs[i] = nbeta, nalpha

  # Assign each electron to an atom initially.
  electron_positions = []
  for i in range(2):
    for j in range(len(molecule)):
      atom_position = jnp.asarray(molecule[j].coords)
      electron_positions.append(
          jnp.tile(atom_position, atomic_spin_configs[j][i]))
  electron_positions = jnp.concatenate(electron_positions)
  # Create a batch of configurations with a Gaussian distribution about each
  # atom.
  key, subkey = jax.random.split(key)
  return (
      electron_positions +
      jax.random.normal(subkey, shape=(batch_size, electron_positions.size)))


from kfac_ferminet_alpha import loss_functions

def make_loss(network, batch_network, atoms, charges, clip_local_energy=0.0):
  """Creates the loss function, including custom gradients.

  Args:
    network: function, signature (params, data), which evaluates the log of
      the magnitude of the wavefunction (square root of the log probability
      distribution) at the single MCMC configuration in data given the network
      parameters.
    batch_network: as for network but data is a batch of MCMC configurations.
    atoms: array of (natoms, ndim) specifying the positions of the nuclei.
    charges: array of (natoms) specifying the nuclear charges.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.

  Returns:
    Callable with signature (params, data) and returns (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
  el_fun = hamiltonian.local_energy(network, atoms, charges)
  batch_local_energy = jax.vmap(el_fun, in_axes=(None, 0), out_axes=0)

  @jax.custom_jvp
  def total_energy(params, data):
    """Evaluates the total energy of the network for a batch of configurations.

    Args:
      params: parameters to pass to the network.
      data: (batched) MCMC configurations to pass to the network.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
    e_l = batch_local_energy(params, data)
    loss = constants.pmean(jnp.mean(e_l))
    variance = constants.pmean(jnp.mean((e_l - loss)**2))
    return loss, AuxiliaryLossData(variance=variance, local_energy=e_l)

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    params, data = primals
    loss, aux_data = total_energy(params, data)

    if clip_local_energy > 0.0:
      # Try centering the window around the median instead of the mean?
      tv = jnp.mean(jnp.abs(aux_data.local_energy - loss))
      tv = constants.pmean(tv)
      diff = jnp.clip(aux_data.local_energy,
                      loss - clip_local_energy*tv,
                      loss + clip_local_energy*tv) - loss
    else:
      diff = aux_data.local_energy - loss

    psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
    loss_functions.register_normal_predictive_distribution(psi_primal[:, None])
    primals_out = loss, aux_data
    tangents_out = (jnp.dot(psi_tangent, diff), aux_data)
    return primals_out, tangents_out

  return total_energy
