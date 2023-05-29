
# from kfac_ferminet_alpha import curvature_blocks
import test_curvature_blocks as curvature_blocks
from kfac_ferminet_alpha import tracer
from kfac_ferminet_alpha import utils

import jax
import jax.numpy as jnp
import numpy as np
import collections

@utils.Stateful.infer_class_state
class CurvatureEstimator(utils.Stateful):
    """
    """
    blocks: None
    damping: None
    
    def __init__(self, tagged_func, func_args, l2_reg, estimation_mode, params_index=0, layer_tag_to_block_cls=None):
        if estimation_mode not in ("fisher_gradients", "fisher_empirical",
                               "fisher_exact", "fisher_curvature_prop",
                               "ggn_exact", "ggn_curvature_prop"):
            raise ValueError(f"Unrecognised estimation_mode={estimation_mode}.")
        super().__init__()
        self.tagged_func = tagged_func
        self.l2_reg = l2_reg
        self.estimation_mode = estimation_mode
        self.params_index = params_index
        self.vjp = tracer.trace_estimator_vjp(self.tagged_func)

        # Figure out the mapping from layer
        self.layer_tag_to_block_cls = curvature_blocks.copy_default_tag_to_block()
        if layer_tag_to_block_cls is None:
            layer_tag_to_block_cls = dict()
        layer_tag_to_block_cls = dict(**layer_tag_to_block_cls)
        self.layer_tag_to_block_cls.update(layer_tag_to_block_cls)

        # Create the blocks
        self._in_tree = jax.tree_structure(func_args)
        self._jaxpr = jax.make_jaxpr(self.tagged_func)(*func_args).jaxpr
        self._layer_tags, self._loss_tags = tracer.extract_tags(self._jaxpr)
        self.blocks = collections.OrderedDict()
        counters = dict()
        for eqn in self._layer_tags:
            cls = self.layer_tag_to_block_cls[eqn.primitive.name]
            c = counters.get(cls.__name__, 0)
            self.blocks[cls.__name__ + "_" + str(c)] = cls(eqn)
            counters[cls.__name__] = c + 1

    def init(self, rng, init_damping):
        return dict(
            blocks=collections.OrderedDict(
            (name, block.init(block_rng))  #
            for (name, block), block_rng  #
            in zip(self.blocks.items(), jax.random.split(rng, len(self.blocks)))),
        damping=init_damping)

    @property
    def diagonal_weight(self):
        return self.l2_reg + self.damping

    @property
    def mat_type(self) -> str:
        return self.estimation_mode.split("_")[0]

    def vectors_to_blocks(self, parameter_structured_vector):
        """
        Splits the parameters to values for the corresponding blocks
        """
        in_vars = jax.tree_unflatten(self._in_tree, self._jaxpr.invars)
        params_vars = in_vars[self.params_index]
        params_vars_flat = jax.tree_flatten(params_vars)[0]
        params_values_flat = jax.tree_flatten(parameter_structured_vector)[0]
        assert len(params_vars_flat) == len(params_values_flat)
        params_dict = dict(zip(params_vars_flat, params_values_flat))
        per_block_vectors = []
        for eqn in self._layer_tags:
            if eqn.primitive.name == "generic_tag":
                block_vars = eqn.invars
            else:
                block_vars = eqn.primitive.split_all_inputs(eqn.invars)[2]
            per_block_vectors.append(tuple(params_dict.pop(v) for v in block_vars))
        if params_dict:
            raise ValueError(f"From the parameters the following structure is not "
                            f"assigned to any block: {params_dict}. Most likely "
                            f"this part of the parameters is not part of the graph "
                            f"reaching the losses.")
        return tuple(per_block_vectors)

    def blocks_to_vectors(self, per_block_vectors):
        """
        Reverses the function self.vectors_to_blocks
        """
        in_vars = jax.tree_unflatten(self._in_tree, self._jaxpr.invars)
        params_vars = in_vars[self.params_index]
        assigned_dict = dict()
        for eqn, block_values in zip(self._layer_tags, per_block_vectors):
            if eqn.primitive.name == "generic_tag":
                block_params = eqn.invars
            else:
                block_params = eqn.primitive.split_all_inputs(eqn.invars)[2]
            assigned_dict.update(zip(block_params, block_values))
        params_vars_flat, params_tree = jax.tree_flatten(params_vars)
        params_values_flat = [assigned_dict[v] for v in params_vars_flat]
        assert len(params_vars_flat) == len(params_values_flat)
        return jax.tree_unflatten(params_tree, params_values_flat)

    def vec_block_apply(self, func, parameter_structured_vector):
        per_block_vectors = self.vectors_to_blocks(parameter_structured_vector)
        assert len(per_block_vectors) == len(self.blocks)
        results = jax.tree_multimap(func, tuple(self.blocks.values()), per_block_vectors)
        parameter_structured_result = self.blocks_to_vectors(results)
        utils.check_structure_shapes_and_dtype(parameter_structured_vector, parameter_structured_result)
        return parameter_structured_result

    def multiply_matpower(self, parameter_structured_vector, exp):
        """
        Multiplies the vectors by the corresponding matrix powers of the blocks
        """
        
        def func(block, vec):
            return block.multiply_matpower(vec, exp, self.diagonal_weight)
        
        return self.vec_block_apply(func, parameter_structured_vector)

    def multiply(self, parameter_structured_vector):
        return self.multiply_matpower(parameter_structured_vector, 1)

    def update_curvature_matrix_estimate(self, ema_old, ema_new, batch_size, rng, func_args, pmap_axis_name):
        # Compute the losses and the VJP function from the function inputs
        losses, losses_vjp = self.vjp(func_args)    # losses: [batchsize/num_device, 1]

        # Helper function that updates the blocks given a vjp vector
        def _update_blocks(vjp_vec_, ema_old_, ema_new_): # vjp_vec_: loss [batch_size, 1]
            blocks_info_ = losses_vjp(vjp_vec_) # w and b; here blocks_info_ may be fisher matrix
            for block_, block_info_ in zip(self.blocks.values(), blocks_info_):
                block_.update_curvature_matrix_estimate(
                    info=block_info_,
                    batch_size=batch_size,
                    ema_old=ema_old_,
                    ema_new=ema_new_,
                    pmap_axis_name=pmap_axis_name)

        if  self.estimation_mode in ("fisher_exact", "ggn_exact"):
            zero_tangents = jax.tree_map(jnp.zeros_like,
                                   list(loss.inputs for loss in losses))
        if self.estimation_mode == "fisher_exact":
            num_indices = [
                (l, int(np.prod(l.fisher_factor_inner_shape[1:]))) for l in losses
            ]
        else:
            num_indices = [
                (l, int(np.prod(l.ggn_factor_inner_shape()))) for l in losses
            ]
        total_num_indices = sum(n for _, n in num_indices)
        for i, (loss, loss_num_indices) in enumerate(num_indices):
            for index in range(loss_num_indices):
                vjp_vec = zero_tangents.copy()
                if self.estimation_mode == "fisher_exact":
                    vjp_vec[i] = loss.multiply_fisher_factor_replicated_one_hot([index])
                else:
                    vjp_vec[i] = loss.multiply_ggn_factor_replicated_one_hot([index])
                if isinstance(vjp_vec[i], jnp.ndarray):
                    # In the special case of only one parameter, it still needs to be a
                    # tuple for the tangents.
                    vjp_vec[i] = (vjp_vec[i],)
                vjp_vec[i] = jax.tree_map(lambda x: x * total_num_indices, vjp_vec[i])
                _update_blocks(tuple(vjp_vec), ema_old, ema_new / total_num_indices)
                ema_old = 1.0

    def update_curvature_estimate_inverse(self, pmap_axis_name, state):
        if state is not None:
            old_state = self.get_state()
        self.set_state(state)
        for block in self.blocks.values():
            block.update_curvature_inverse_estimate(self.diagonal_weight,
                                              pmap_axis_name)
        if state is None:
            return None
        else:
            state = self.pop_state()
            self.set_state(old_state)
            return state
