
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jnr

# from kfac_ferminet_alpha import estimator
import test_estimator as estimator
from kfac_ferminet_alpha import tag_graph_matcher as tgm
from kfac_ferminet_alpha import utils


@utils.Stateful.infer_class_state
class Optimizer(utils.Stateful):
  """The default optimizer class."""
  velocities: None
  estimator: estimator.CurvatureEstimator
  step_counter: jnp.ndarray

  def __init__(self,
      value_and_grad_func,
      l2_reg,
      value_func_has_aux = False,
      value_func_has_state = False,
      value_func_has_rng = False,
      learning_rate_schedule = None,
      momentum_schedule = None,
      damping_schedule = None,
      min_damping = 1e-8,
      max_damping = jnp.inf,
      norm_constraint = None,
      num_burnin_steps = 10,
      estimation_mode = "fisher_gradients",
      curvature_ema = 0.95,
      inverse_update_period = 5,
      register_only_generic = False,
      layer_tag_to_block_cls = None,
      patterns_to_skip = (),
      donate_parameters = False,
      donate_optimizer_state = False,
      donate_batch_inputs = False,
      donate_func_state = False,
      batch_process_func = None,
      multi_device = False,
      use_jax_cond = True,
      debug: bool = False,
      pmap_axis_name="kfac_axis",
  ):
    super().__init__()
    self.value_and_grad_func = value_and_grad_func
    self.value_func_has_aux = value_func_has_aux
    self.value_func_has_state = value_func_has_state
    self.value_func_has_rng = value_func_has_rng
    self.value_func = utils.convert_value_and_grad_to_value_func(
        value_and_grad_func, has_aux=value_func_has_aux)
    self.l2_reg = l2_reg
    self.learning_rate_schedule = learning_rate_schedule
    if momentum_schedule is not None:

      def schedule_with_first_step_zero(global_step: jnp.ndarray):
        value = momentum_schedule(global_step)
        check = jnp.equal(global_step, 0)
        return check * jnp.zeros_like(value) + (1 - check) * value

      self.momentum_schedule = schedule_with_first_step_zero
    else:
      self.momentum_schedule = None
    self.damping_schedule = damping_schedule
    self.min_damping = min_damping
    self.max_damping = max_damping
    self.norm_constraint = norm_constraint
    self.num_burnin_steps = num_burnin_steps
    self.estimation_mode = estimation_mode
    self.curvature_ema = curvature_ema
    self.inverse_update_period = inverse_update_period
    self.register_only_generic = register_only_generic
    self.layer_tag_to_block_cls = layer_tag_to_block_cls
    self.patterns_to_skip = patterns_to_skip
    self.donate_parameters = donate_parameters
    self.donate_optimizer_state = donate_optimizer_state
    self.donate_batch_inputs = donate_batch_inputs
    self.donate_func_state = donate_func_state
    self.batch_process_func = batch_process_func or (lambda x: x)
    self.multi_device = multi_device
    self.use_jax_cond = use_jax_cond
    self.debug = debug
    self.pmap_axis_name = pmap_axis_name if multi_device else None
    self._rng_split = utils.p_split if multi_device else jnr.split

    # Attributes filled in during self.init()
    self.finalized = False
    self.tagged_func = None
    self.flat_params_shapes = None
    self.params_treedef = None
    # Special attributes related to jitting/pmap
    self._jit_init = None
    self._jit_burnin = None
    self._jit_step = None

  def finalize(self, params, rng, batch, func_state = None):
    """
    Finalizes the optimizer by tracing the model function with the params and batch.
    """
    if self.finalized:
      raise ValueError("Optimizer has already been finalized.")
    if self.multi_device:
      # We assume that the parameters and batch are replicated, while tracing
      # must happen with parameters for a single device call
      params, rng, batch = jax.tree_map(lambda x: x[0], (params, rng, batch))
      if func_state is not None:
        func_state = jax.tree_map(lambda x: x[0], func_state)
    batch = self.batch_process_func(batch)
    # These are all tracing operations and we can run them with abstract values
    func_args = utils.make_func_args(params, func_state, rng, batch,
                                     self.value_func_has_state,
                                     self.value_func_has_rng)
    # Run all tracing with abstract values so no computation is done
    flat_params, self.params_treedef = jax.tree_flatten(params)
    self.flat_params_shapes = tuple(p.shape for p in flat_params)
    self.tagged_func = tgm.auto_register_tags(
        func=self.value_func,
        func_args=func_args,
        params_index=0,
        register_only_generic=self.register_only_generic,
        patterns_to_skip=self.patterns_to_skip)
    self.estimator = estimator.CurvatureEstimator(
        self.tagged_func,
        func_args,
        self.l2_reg,
        self.estimation_mode,
        layer_tag_to_block_cls=self.layer_tag_to_block_cls)
    # Arguments: params, opt_state, rng, batch, func_state
    donate_argnums = []
    if self.donate_parameters:
      donate_argnums.append(0)
    if self.donate_optimizer_state:
      donate_argnums.append(1)
    if self.donate_batch_inputs:
      donate_argnums.append(3)
    if self.donate_func_state and self.value_func_has_state:
      donate_argnums.append(4)
    donate_argnums = tuple(donate_argnums)

    if self.debug:
      self._jit_init = self._init
      self._jit_burnin = self._burnin
      self._jit_step = self._step
    elif self.multi_device:
      self._jit_init = jax.pmap(
          self._init, axis_name=self.pmap_axis_name, donate_argnums=[0])
      # batch size is static argnum and is at index 5
      self._jit_burnin = jax.pmap(
          self._burnin,
          axis_name=self.pmap_axis_name,
          static_broadcasted_argnums=[5])
      self._jit_step = jax.pmap(
          self._step,
          axis_name=self.pmap_axis_name,
          donate_argnums=donate_argnums,
          static_broadcasted_argnums=[5])
    else:
      self._jit_init = jax.jit(self._init, donate_argnums=[0])
      # batch size is static argnum and is at index 5
      self._jit_burnin = jax.jit(self._burnin, static_argnums=[5])
      self._jit_step = jax.jit(
          self._step, donate_argnums=donate_argnums, static_argnums=[5])
    self.finalized = True

  def _init(self, rng):
    """This is the non-jitted version of initializing the state."""
    flat_velocities = [jnp.zeros(shape) for shape in self.flat_params_shapes]
    return dict(
        velocities=jax.tree_unflatten(self.params_treedef, flat_velocities),
        estimator=self.estimator.init(rng, None),
        step_counter=jnp.asarray(0))

  def verify_args_and_get_step_counter(
      self,
      params,
      state,
      rng,
      data_iterator,
      func_state,
      learning_rate = None,
      momentum = None,
      damping = None,
      global_step_int = None,
  ):
    """Verifies that the arguments passed to `Optimizer.step` are correct."""
    if not self.finalized:
      rng, rng_finalize = self._rng_split(rng)
      self.finalize(params, rng_finalize, next(data_iterator), func_state)
    # Verify correct arguments invocation
    if self.learning_rate_schedule is not None and learning_rate is not None:
      raise ValueError("When you have passed a `learning_rate_schedule` you "
                       "should not pass a value to the step function.")
    if self.momentum_schedule is not None and momentum is not None:
      raise ValueError("When you have passed a `momentum_schedule` you should "
                       "not pass a value to the step function.")
    if self.damping_schedule is not None and damping is not None:
      raise ValueError("When you have passed a `damping_schedule` you should "
                       "not pass a value to the step function.")
    # Do a bunrnin on the first iteration
    if global_step_int is None:
      if self.multi_device:
        return int(utils.get_first(state["step_counter"]))
      else:
        return int(state["step_counter"])
    return global_step_int

  def _burnin(
      self,
      params,
      state,
      rng,
      batch,
      func_state,
      batch_size,
  ):
    """This is the non-jitted version of a single burnin step."""
    self.set_state(state)
    batch = self.batch_process_func(batch)
    rng, func_rng = jnr.split(rng) if self.value_func_has_rng else (rng, None)
    func_args = utils.make_func_args(params, func_state, func_rng, batch,
                                     self.value_func_has_state,
                                     self.value_func_has_rng)

    # Compute batch size
    if batch_size is None:
      batch_size = jax.tree_flatten(batch)[0][0].shape[0]

    # Update curvature estimate
    ema_old, ema_new = 1.0, 1.0 / self.num_burnin_steps
    self.estimator.update_curvature_matrix_estimate(ema_old, ema_new,
                                                    batch_size, rng, func_args,
                                                    self.pmap_axis_name)

    if func_state is not None:
      out, _ = self.value_and_grad_func(*func_args)
      _, func_state, _ = utils.extract_func_outputs(out,
                                                    self.value_func_has_aux,
                                                    self.value_func_has_state)

    return self.pop_state(), func_state

  def step(
      self,
      params,
      state,
      rng,
      data_iterator,
      func_state = None,
      learning_rate = None,
      momentum = None,
      damping = None,
      batch_size = None,
      global_step_int = None,
  ) :
    """Performs a single update step using the optimizer.

    Args:
      params: The parameters of the model.
      state: The state of the optimizer.
      rng: A Jax PRNG key.
      data_iterator: An iterator that returns a batch of data.
      func_state: Any function state that gets passed in and returned.
      learning_rate: This must be provided when
        `use_adaptive_learning_rate=False` and `learning_rate_schedule=None`.
      momentum: This must be provided when
        `use_adaptive_momentum=False` and `momentum_schedule=None`.
      damping: This must be provided when
        `use_adaptive_damping=False` and `damping_schedule=None`.
      batch_size: The batch size to use for KFAC. The default behaviour when it
        is None is to use the leading dimension of the first data array.
      global_step_int: The global step as a python int. Note that this must
        match the step inte  rnal to the optimizer that is part of its state.

    Returns:
      (params, state, stats)
      where:
          params: The updated model parameters.
          state: The updated optimizer state.
          stats: A dictionary of key statistics provided to be logged.
    """
    step_counter_int = self.verify_args_and_get_step_counter(
        params=params,
        state=state,
        rng=rng,
        data_iterator=data_iterator,
        func_state=func_state,
        learning_rate=learning_rate,
        momentum=momentum,
        damping=damping,
        global_step_int=global_step_int)

    if step_counter_int == 0:
      for _ in range(self.num_burnin_steps):
        rng, rng_burn = self._rng_split(rng)
        batch = next(data_iterator)
        state, func_state = self._jit_burnin(params, state, rng_burn, batch,
                                             func_state, batch_size)

      # On the first step we always treat the momentum as 0.0
      if self.momentum_schedule is None:
        momentum = jnp.zeros([])
        if self.multi_device:
          momentum = utils.replicate_all_local_devices(momentum)

    batch = next(data_iterator)
    return self._jit_step(params, state, rng, batch, func_state, batch_size,
                          learning_rate, momentum, damping)

  def _step(
      self,
      params,
      state,
      rng,
      batch,
      func_state,
      batch_size,
      learning_rate,
      momentum,
      damping,
  ):
    """This is the non-jitted version of a single step."""
    # Unpack and set the state
    self.set_state(state)
    if damping is not None:
      assert self.estimator.damping is None
      self.estimator.damping = damping
    else:
      assert self.estimator.damping is not None

    # Preprocess the batch and construct correctly the function arguments
    batch = self.batch_process_func(batch)
    rng, func_rng = jnr.split(rng) if self.value_func_has_rng else (rng, None)
    func_args = utils.make_func_args(params, func_state, func_rng, batch,
                                     self.value_func_has_state,
                                     self.value_func_has_rng)

    # Compute the batch size
    if batch_size is None:
      batch_size = jax.tree_flatten(batch)[0][0].shape[0]

    # Compute schedules if applicable
    if self.learning_rate_schedule is not None:
      assert learning_rate is None
      learning_rate = self.learning_rate_schedule(self.step_counter)
    else:
      assert learning_rate is not None
    if self.momentum_schedule is not None:
      assert momentum is None
      momentum = self.momentum_schedule(self.step_counter)
    else:
      assert momentum is not None
    if self.damping_schedule is not None:
      assert damping is None
      damping = self.damping_schedule(self.step_counter)
    else:
      assert damping is not None

    # Compute current loss and gradients
    out, grads = self.value_and_grad_func(*func_args)
    loss, new_func_state, aux = utils.extract_func_outputs(
        out, self.value_func_has_aux, self.value_func_has_state)
    # Sync loss and grads
    loss, grads = utils.pmean_if_pmap((loss, grads), self.pmap_axis_name)

    # Update curvature estimate # calculate xxt and ggt
    self.estimator.update_curvature_matrix_estimate(
        self.curvature_ema,
        1.0,
        batch_size,
        rng,
        func_args,
        self.pmap_axis_name,
    )

    # Optionally update the inverse estimate
    self.estimator.set_state(
        lax.cond(
            self.step_counter % self.inverse_update_period == 0,
            lambda s: self.estimator.update_curvature_estimate_inverse(  # pylint: disable=g-long-lambda
                self.pmap_axis_name, s),
            lambda s: s,
            self.estimator.pop_state()))

    # Compute proposed directions.  Preconditioned grads
    vectors = self.propose_directions(
        grads,
        self.velocities,
        learning_rate,
        momentum,
    )

    # The learning rate is defined as the negative of the coefficient by which
    # we multiply the gradients, while the momentum is the coefficient by
    # which we multiply the velocities.
    neg_learning_rate = -learning_rate
    # Compute the coefficients of the update vectors
    assert neg_learning_rate is not None and momentum is not None
    coefficients = (neg_learning_rate, momentum)

    # Update velocities and compute new delta
    self.velocities, delta = self.velocities_and_delta(
        self.velocities,
        vectors,
        coefficients,
    )

    # Update parameters: params = params + delta
    params = jax.tree_multimap(jnp.add, params, delta)

    # Optionally compute the reduction ratio and update the damping
    self.estimator.damping = None
    rho = jnp.nan

    # Statistics with useful information
    stats = dict()
    stats["step"] = self.step_counter
    stats["loss"] = loss
    stats["learning_rate"] = -coefficients[0]
    stats["momentum"] = coefficients[1]
    stats["damping"] = damping
    stats["rho"] = rho
    if self.value_func_has_aux:
      stats["aux"] = aux
    self.step_counter = self.step_counter + 1

    if self.value_func_has_state:
      return params, self.pop_state(), new_func_state, stats
    else:
      assert new_func_state is None
      return params, self.pop_state(), stats

  def init(
      self,
      params,
      rng,
      batch,
      func_state = None,
  ):
    """Initializes the optimizer and returns the appropriate optimizer state."""
    if not self.finalized:
      self.finalize(params, rng, batch, func_state)
    return self._jit_init(rng)

  def propose_directions(
      self,
      grads,
      velocities,
      learning_rate,
      momentum,
  ):
    """Computes the vector proposals for the next step."""
    del momentum  # not used in this, but could be used in subclasses
    preconditioned_grads = self.estimator.multiply_matpower(grads, -1)

    if self.norm_constraint is not None:
      assert learning_rate is not None
      sq_norm_grads = utils.inner_product(preconditioned_grads, grads)
      sq_norm_scaled_grads = sq_norm_grads * learning_rate**2

      # We need to sync the norms here, because reduction can be
      # non-deterministic. They specifically are on GPUs by default for better
      # performance. Hence although grads and preconditioned_grads are synced,
      # the inner_product operation can still produce different answers on
      # different devices.
      sq_norm_scaled_grads = utils.pmean_if_pmap(sq_norm_scaled_grads,
                                                 self.pmap_axis_name)

      max_coefficient = jnp.sqrt(self.norm_constraint / sq_norm_scaled_grads)
      coefficient = jnp.minimum(max_coefficient, 1)
      preconditioned_grads = utils.scalar_mul(preconditioned_grads, coefficient)

    return preconditioned_grads, velocities

  def velocities_and_delta(
      self,
      velocities,
      vectors,
      coefficients,
  ):
    """Computes the new velocities and delta (update to parameters)."""
    del velocities
    assert len(vectors) == len(coefficients)
    delta = utils.scalar_mul(vectors[0], coefficients[0])
    for vi, wi in zip(vectors[1:], coefficients[1:]):
      delta = jax.tree_multimap(jnp.add, delta, utils.scalar_mul(vi, wi))
    return delta, delta
