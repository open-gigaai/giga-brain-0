import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Keep these groups aligned with EmbodimentId in giga_brain_0_transforms.py.
_FAST_TOKEN_DOMAIN_EMBODIMENT_IDS = {
    'robot': (0, 1, 2, 6, 7),
    'umi': (3,),
    'ego': (4, 5),
}


class GigaBrain07Loss(nn.Module):
    """Loss function for GigaBrain07, including diffusion loss for actions,
    cross-entropy loss for language tokens, and MSE loss for trajectory
    prediction.
    """

    def __init__(
        self,
        llm_loss_weight: float = 1.0,
        flow_action_dim_loss_weight_cfg: dict[str, object] | None = None,
    ):
        """Initializes the GigaBrain07Loss module."""
        super().__init__()
        self.u_t: torch.Tensor | None = None

        self.llm_loss_weight = float(llm_loss_weight)
        if self.llm_loss_weight < 0.0:
            raise ValueError(f'llm_loss_weight must be greater than or equal to 0, but got {self.llm_loss_weight}')

        self.flow_action_dim_loss_weights = self._parse_flow_action_dim_loss_weight_cfg(
            flow_action_dim_loss_weight_cfg
        )

    @staticmethod
    def _parse_flow_action_dim_loss_weight_cfg(
        cfg: dict[str, object] | None,
    ) -> dict[int, tuple[float, ...]]:
        cfg = dict(cfg or {})
        if not bool(cfg.get('enabled', bool(cfg))):
            return {}

        unknown_keys = set(cfg) - {'enabled', 'weights_by_embodiment'}
        if unknown_keys:
            raise ValueError(
                'Unknown flow_action_dim_loss_weight_cfg keys: '
                f'{sorted(unknown_keys)}'
            )
        weights_by_embodiment = cfg.get('weights_by_embodiment')
        if not isinstance(weights_by_embodiment, dict) or not weights_by_embodiment:
            raise ValueError(
                'flow_action_dim_loss_weight_cfg.weights_by_embodiment must be a non-empty dict'
            )

        parsed: dict[int, tuple[float, ...]] = {}
        for raw_embodiment_id, raw_rules in weights_by_embodiment.items():
            embodiment_id = int(raw_embodiment_id)
            if embodiment_id < 0:
                raise ValueError(f'embodiment id must be non-negative, got {embodiment_id}')
            if not isinstance(raw_rules, (list, tuple)) or not raw_rules:
                raise ValueError(
                    f'flow loss weight rules for embodiment {embodiment_id} must be a non-empty list'
                )

            dim_weights = [1.0] * 32
            configured_dims: set[int] = set()
            for raw_rule in raw_rules:
                if not isinstance(raw_rule, dict):
                    raise ValueError(
                        f'flow loss weight rule for embodiment {embodiment_id} must be a dict'
                    )
                rule = dict(raw_rule)
                unknown_rule_keys = set(rule) - {'dims', 'weight'}
                if unknown_rule_keys:
                    raise ValueError(
                        f'Unknown flow loss weight rule keys for embodiment {embodiment_id}: '
                        f'{sorted(unknown_rule_keys)}'
                    )
                dims = rule.get('dims')
                if not isinstance(dims, (list, tuple)) or not dims:
                    raise ValueError(
                        f'flow loss weight rule dims for embodiment {embodiment_id} must be non-empty'
                    )
                weight = float(rule.get('weight', float('nan')))
                if not math.isfinite(weight) or weight < 0.0:
                    raise ValueError(
                        f'flow loss weight for embodiment {embodiment_id} must be finite and non-negative, got {weight}'
                    )
                for raw_dim in dims:
                    dim = int(raw_dim)
                    if not 0 <= dim < 32:
                        raise ValueError(
                            f'flow loss dimension for embodiment {embodiment_id} must be in [0, 31], got {dim}'
                        )
                    if dim in configured_dims:
                        raise ValueError(
                            f'flow loss dimension {dim} is configured more than once for embodiment {embodiment_id}'
                        )
                    configured_dims.add(dim)
                    dim_weights[dim] = weight
            parsed[embodiment_id] = tuple(dim_weights)
        return parsed

    def _flow_action_dim_loss_weight(
        self,
        pred_velocity: torch.Tensor,
        embodiment_id: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.flow_action_dim_loss_weights:
            return None
        if embodiment_id is None:
            raise ValueError(
                'embodiment_id is required when flow_action_dim_loss_weight_cfg is enabled'
            )

        embodiment_id = embodiment_id.to(device=pred_velocity.device).long().view(-1)
        if embodiment_id.shape[0] != pred_velocity.shape[0]:
            raise ValueError(
                f'embodiment_id batch size {embodiment_id.shape[0]} does not match '
                f'prediction batch size {pred_velocity.shape[0]}'
            )

        sample_weights = pred_velocity.new_ones((pred_velocity.shape[0], 32))
        for emb_idx, dim_weights in self.flow_action_dim_loss_weights.items():
            configured_weights = pred_velocity.new_tensor(dim_weights)
            sample_weights = torch.where(
                (embodiment_id == emb_idx)[:, None],
                configured_weights[None, :],
                sample_weights,
            )
        return sample_weights[:, None, :]

    def sample_noise(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Samples Gaussian noise.

        Args:
            shape (tuple[int, ...]): The shape of the noise tensor.
            device (torch.device): The device to place the tensor on.

        Returns:
            torch.Tensor: The sampled noise.
        """
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def _sample_beta(self, alpha: float, beta: float, bsize: int, device: torch.device) -> torch.Tensor:
        """Samples from a Beta distribution.

        Args:
            alpha (float): The alpha parameter of the Beta distribution.
            beta (float): The beta parameter of the Beta distribution.
            bsize (int): The batch size.
            device (torch.device): The device to place the tensors on.

        Returns:
            torch.Tensor: Samples from the Beta distribution.
        """
        alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32, device=device)
        beta_tensor = torch.as_tensor(beta, dtype=torch.float32, device=device)
        return torch.distributions.Beta(alpha_tensor, beta_tensor).sample((bsize,))

    def sample_time(self, bsize: int, device: torch.device) -> torch.Tensor:
        """Samples timesteps for the diffusion process.

        Args:
            bsize (int): The batch size.
            device (torch.device): The device to place the tensor on.

        Returns:
            torch.Tensor: The sampled timesteps.
        """
        time_beta = self._sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def add_noise(
        self,
        actions: torch.Tensor,
        action_dim_loss_mask: torch.Tensor | None = None,
        batch_mul: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adds noise to actions for the diffusion process and stores
        intermediate values.

        Args:
            actions (torch.Tensor): The original actions.
            action_dim_loss_mask (torch.Tensor | None): Optional valid-action
                mask with shape [B, T, 32]. Invalid dimensions remain zero in
                both the noisy model input and flow target.
            batch_mul (int): Number of independent flow-matching samples per
                action target. Defaults to 1.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the noisy actions and the timesteps.
        """
        if actions.ndim != 3 or actions.shape[-1] != 32:
            raise ValueError(f'actions must have shape [B, T, 32], got {tuple(actions.shape)}')
        if batch_mul < 1:
            raise ValueError(f'batch_mul must be positive, but got {batch_mul}')
        if batch_mul > 1:
            actions = actions.repeat_interleave(batch_mul, dim=0)
            if action_dim_loss_mask is not None:
                action_dim_loss_mask = action_dim_loss_mask.repeat_interleave(batch_mul, dim=0)

        noise = self.sample_noise(actions.shape, actions.device)
        if action_dim_loss_mask is not None:
            if action_dim_loss_mask.shape != actions.shape:
                raise ValueError(
                    f'action_dim_loss_mask shape {tuple(action_dim_loss_mask.shape)} does not match '
                    f'action shape {tuple(actions.shape)}'
                )
            dim_mask = action_dim_loss_mask.to(device=actions.device, dtype=actions.dtype)
            actions = actions * dim_mask
            noise = noise * dim_mask
        time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        self.u_t = u_t

        return x_t, time

    @staticmethod
    def _reduce_masked_token_loss(token_loss: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        token_mask = token_mask.to(device=token_loss.device, dtype=token_loss.dtype)
        return (token_loss * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp(min=1.0)

    @staticmethod
    def _mean_active_sample_loss(per_sample_loss: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        active_mask = active_mask.to(device=per_sample_loss.device, dtype=torch.bool)
        if active_mask.any():
            return per_sample_loss[active_mask].mean()
        return per_sample_loss.sum() * 0.0

    def llm_loss_terms(
        self,
        logits: torch.Tensor,
        gt_lang_tokens: torch.Tensor,
        lang_loss_masks: torch.Tensor,
        fast_action_indicator: torch.Tensor | None = None,
        subtask_indicator: torch.Tensor | None = None,
        logits_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calculates per-sample language losses, optionally split by task.

        Args:
            logits (torch.Tensor): The predicted logits from the language model.
            gt_lang_tokens (torch.Tensor): The ground truth language tokens.
            lang_loss_masks (torch.Tensor): The mask to apply to the language loss.
            fast_action_indicator (torch.Tensor | None): Optional mask for FAST
                action tokens aligned with the original language sequence.
            subtask_indicator (torch.Tensor | None): Optional mask for subtask
                text tokens aligned with the original language sequence.

        Returns:
            dict[str, torch.Tensor]: Per-sample all-token language loss,
                generic non-split language loss, plus optional split losses
                for FAST action tokens and subtask tokens.
        """
        gt_lang_tokens = gt_lang_tokens[:, 1:]
        lang_loss_masks = lang_loss_masks[:, 1:]
        if logits_mask is None:
            logits = logits[:, :-1, :]  # remove the last token <eos>
            vocab_size = logits.shape[-1]
            token_loss = F.cross_entropy(logits.reshape(-1, vocab_size), gt_lang_tokens.reshape(-1).long(), reduction='none')
            token_loss = token_loss.reshape(gt_lang_tokens.shape[0], gt_lang_tokens.shape[1])
        else:
            logits_mask = logits_mask.to(device=gt_lang_tokens.device, dtype=torch.bool)
            if logits_mask.shape != gt_lang_tokens.shape:
                raise ValueError(
                    f'logits_mask shape {tuple(logits_mask.shape)} does not match shifted language tokens shape {tuple(gt_lang_tokens.shape)}'
                )
            if logits.ndim != 2:
                raise ValueError(f'compact logits must have shape [N, vocab], got {tuple(logits.shape)}')

            compact_targets = gt_lang_tokens[logits_mask].long()
            compact_loss = F.cross_entropy(logits, compact_targets, reduction='none')
            token_loss = compact_loss.new_zeros(gt_lang_tokens.shape)
            token_loss[logits_mask.to(device=token_loss.device)] = compact_loss

        shifted_lang_loss_masks = lang_loss_masks.to(device=token_loss.device, dtype=token_loss.dtype)
        split_token_mask = torch.zeros_like(shifted_lang_loss_masks, dtype=torch.bool)
        loss_terms = {
            'total_llm_loss': self.llm_loss_weight * self._reduce_masked_token_loss(token_loss, shifted_lang_loss_masks)
        }

        if fast_action_indicator is not None:
            shifted_fast_mask = fast_action_indicator[:, 1:]
            if shifted_fast_mask.shape != gt_lang_tokens.shape:
                raise ValueError(
                    f'fast_action_indicator shape {tuple(shifted_fast_mask.shape)} does not match shifted language tokens shape {tuple(gt_lang_tokens.shape)}'
                )
            fast_token_mask = shifted_lang_loss_masks * shifted_fast_mask.to(dtype=shifted_lang_loss_masks.dtype)
            split_token_mask |= fast_token_mask.to(dtype=torch.bool)
            loss_terms['fast_token_llm_loss'] = self.llm_loss_weight * self._reduce_masked_token_loss(
                token_loss,
                fast_token_mask,
            )

        if subtask_indicator is not None:
            shifted_subtask_mask = subtask_indicator[:, 1:]
            if shifted_subtask_mask.shape != gt_lang_tokens.shape:
                raise ValueError(
                    f'subtask_indicator shape {tuple(shifted_subtask_mask.shape)} does not match shifted language tokens shape {tuple(gt_lang_tokens.shape)}'
                )
            subtask_token_mask = shifted_lang_loss_masks * shifted_subtask_mask.to(dtype=shifted_lang_loss_masks.dtype)
            split_token_mask |= subtask_token_mask.to(dtype=torch.bool)
            loss_terms['subtask_llm_loss'] = self.llm_loss_weight * self._reduce_masked_token_loss(
                token_loss,
                subtask_token_mask,
            )

        generic_token_mask = shifted_lang_loss_masks * (~split_token_mask).to(dtype=shifted_lang_loss_masks.dtype)
        loss_terms['generic_llm_loss'] = self.llm_loss_weight * self._reduce_masked_token_loss(
            token_loss,
            generic_token_mask,
        )

        return loss_terms

    def llm_loss_terms_from_token_loss(
        self,
        token_loss: torch.Tensor,
        lang_loss_masks: torch.Tensor,
        fast_action_indicator: torch.Tensor | None = None,
        subtask_indicator: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calculates per-sample language losses from precomputed shifted token loss."""
        shifted_lang_loss_masks = lang_loss_masks[:, 1:].to(device=token_loss.device, dtype=token_loss.dtype)
        if token_loss.shape != shifted_lang_loss_masks.shape:
            raise ValueError(
                f'token_loss shape {tuple(token_loss.shape)} does not match '
                f'shifted lang_loss_masks shape {tuple(shifted_lang_loss_masks.shape)}'
            )

        split_token_mask = torch.zeros_like(shifted_lang_loss_masks, dtype=torch.bool)
        loss_terms = {
            'total_llm_loss': self.llm_loss_weight * self._reduce_masked_token_loss(token_loss, shifted_lang_loss_masks)
        }

        if fast_action_indicator is not None:
            shifted_fast_mask = fast_action_indicator[:, 1:]
            if shifted_fast_mask.shape != token_loss.shape:
                raise ValueError(
                    f'fast_action_indicator shape {tuple(shifted_fast_mask.shape)} does not match '
                    f'token_loss shape {tuple(token_loss.shape)}'
                )
            fast_token_mask = shifted_lang_loss_masks * shifted_fast_mask.to(dtype=shifted_lang_loss_masks.dtype)
            split_token_mask |= fast_token_mask.to(dtype=torch.bool)
            loss_terms['fast_token_llm_loss'] = self.llm_loss_weight * self._reduce_masked_token_loss(
                token_loss,
                fast_token_mask,
            )

        if subtask_indicator is not None:
            shifted_subtask_mask = subtask_indicator[:, 1:]
            if shifted_subtask_mask.shape != token_loss.shape:
                raise ValueError(
                    f'subtask_indicator shape {tuple(shifted_subtask_mask.shape)} does not match '
                    f'token_loss shape {tuple(token_loss.shape)}'
                )
            subtask_token_mask = shifted_lang_loss_masks * shifted_subtask_mask.to(dtype=shifted_lang_loss_masks.dtype)
            split_token_mask |= subtask_token_mask.to(dtype=torch.bool)
            loss_terms['subtask_llm_loss'] = self.llm_loss_weight * self._reduce_masked_token_loss(
                token_loss,
                subtask_token_mask,
            )

        generic_token_mask = shifted_lang_loss_masks * (~split_token_mask).to(dtype=shifted_lang_loss_masks.dtype)
        loss_terms['generic_llm_loss'] = self.llm_loss_weight * self._reduce_masked_token_loss(
            token_loss,
            generic_token_mask,
        )

        return loss_terms

    def llm_loss(
        self,
        logits: torch.Tensor,
        gt_lang_tokens: torch.Tensor,
        lang_loss_masks: torch.Tensor,
        fast_action_indicator: torch.Tensor | None = None,
        subtask_indicator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculates the total cross-entropy loss for language model predictions."""
        return self.llm_loss_terms(
            logits,
            gt_lang_tokens,
            lang_loss_masks,
            fast_action_indicator=fast_action_indicator,
            subtask_indicator=subtask_indicator,
            logits_mask=None,
        )['total_llm_loss']

    def traj_loss(self, traj_pred: torch.Tensor, gt_traj: torch.Tensor, traj_loss_mask: torch.Tensor) -> torch.Tensor:
        """Calculates the MSE loss for trajectory predictions.

        Args:
            traj_pred (torch.Tensor): The predicted trajectory.
            gt_traj (torch.Tensor): The ground truth trajectory.
            traj_loss_mask (torch.Tensor): The mask to apply to the trajectory loss.

        Returns:
            torch.Tensor: The calculated trajectory loss.
        """
        traj_loss = F.mse_loss(gt_traj, traj_pred, reduction='none')
        traj_loss = traj_loss * traj_loss_mask

        traj_loss = traj_loss.sum(axis=-1) / torch.clamp(traj_loss_mask.sum(axis=-1), min=1)

        return traj_loss.mean(axis=-1)

    def _compute_diffusion_loss(
        self,
        pred_velocity: torch.Tensor,
        action_loss_mask: torch.Tensor,
        action_dim_loss_mask: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred_velocity.ndim != 3 or pred_velocity.shape[-1] != 32:
            raise ValueError(f'pred_velocity must have shape [B, T, 32], got {tuple(pred_velocity.shape)}')
        if self.u_t.shape != pred_velocity.shape:
            raise ValueError(
                f'flow target shape {tuple(self.u_t.shape)} does not match prediction shape {tuple(pred_velocity.shape)}'
            )

        diffusion_loss = F.mse_loss(self.u_t, pred_velocity, reduction='none')

        temporal_mask = action_loss_mask.to(device=pred_velocity.device, dtype=pred_velocity.dtype)
        if temporal_mask.ndim != 2:
            raise ValueError(f'action_loss_mask must have shape [B, T], got {tuple(temporal_mask.shape)}')

        loss_weight = self._flow_action_dim_loss_weight(pred_velocity, embodiment_id)

        if action_dim_loss_mask is not None:
            if action_dim_loss_mask.shape != diffusion_loss.shape:
                raise ValueError(
                    f'action_dim_loss_mask shape {tuple(action_dim_loss_mask.shape)} does not match '
                    f'diffusion loss shape {tuple(diffusion_loss.shape)}'
                )
            dim_mask = action_dim_loss_mask.to(
                device=pred_velocity.device, dtype=pred_velocity.dtype
            )
            dim_mask = dim_mask * temporal_mask[:, :, None]
            valid_dim_count = dim_mask.sum(dim=(1, 2)).clamp(min=1.0)
            if loss_weight is not None:
                diffusion_loss = diffusion_loss * loss_weight
            return (diffusion_loss * dim_mask).sum(dim=(1, 2)) / valid_dim_count

        if loss_weight is not None:
            dim_mask = temporal_mask[:, :, None].expand_as(diffusion_loss)
            valid_dim_count = dim_mask.sum(dim=(1, 2)).clamp(min=1.0)
            return (diffusion_loss * loss_weight * dim_mask).sum(dim=(1, 2)) / valid_dim_count

        per_step_loss = diffusion_loss.mean(axis=-1) * temporal_mask
        valid_step_count = temporal_mask.sum(dim=-1).clamp(min=1.0)
        return per_step_loss.sum(dim=-1) / valid_step_count

    @staticmethod
    def _per_embodiment_diffusion_metrics(
        per_sample_diffusion_loss: torch.Tensor,
        has_action: torch.Tensor,
        embodiment_id: torch.Tensor | None,
        num_embodiments: int,
    ) -> dict[str, torch.Tensor]:
        """Splits the per-sample diffusion loss by embodiment id.

        Each emitted tensor is a scalar; NaN means this rank had no valid
        sample for that embodiment in this step. The trainer is responsible
        for skipping NaNs when reducing across ranks/steps.
        """
        metrics: dict[str, torch.Tensor] = {}
        if embodiment_id is None or num_embodiments <= 0:
            return metrics

        emb = embodiment_id.to(device=per_sample_diffusion_loss.device).long().view(-1)
        if emb.shape[0] != per_sample_diffusion_loss.shape[0]:
            return metrics

        active = has_action.to(device=per_sample_diffusion_loss.device, dtype=torch.bool)
        zero = per_sample_diffusion_loss.sum() * 0.0
        nan = zero + float('nan')
        for emb_idx in range(num_embodiments):
            mask = active & (emb == emb_idx)
            if mask.any():
                metrics[f'metric/diff_loss_emb_{emb_idx}'] = per_sample_diffusion_loss[mask].mean().detach()
            else:
                metrics[f'metric/diff_loss_emb_{emb_idx}'] = nan.detach()
        return metrics

    @staticmethod
    def _per_domain_fast_token_metrics(
        per_sample_fast_token_loss: torch.Tensor,
        fast_active_samples: torch.Tensor,
        embodiment_id: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Splits per-sample FAST-token loss into robot, UMI, and Ego metrics."""
        metrics: dict[str, torch.Tensor] = {}
        if embodiment_id is None:
            return metrics

        emb = embodiment_id.to(device=per_sample_fast_token_loss.device).long().view(-1)
        if emb.shape[0] != per_sample_fast_token_loss.shape[0]:
            return metrics

        active = fast_active_samples.to(
            device=per_sample_fast_token_loss.device,
            dtype=torch.bool,
        )
        zero = per_sample_fast_token_loss.sum() * 0.0
        nan = zero + float('nan')
        for domain, embodiment_ids in _FAST_TOKEN_DOMAIN_EMBODIMENT_IDS.items():
            domain_mask = torch.zeros_like(active)
            for emb_idx in embodiment_ids:
                domain_mask |= emb == emb_idx
            domain_mask &= active
            key = f'metric/fast_token_llm_loss_{domain}'
            if domain_mask.any():
                metrics[key] = per_sample_fast_token_loss[domain_mask].mean().detach()
            else:
                metrics[key] = nan.detach()
        return metrics

    def forward(
        self,
        model_pred: dict[str, torch.Tensor],
        gt_lang_tokens: torch.Tensor,
        lang_loss_masks: torch.Tensor,
        action_loss_mask: torch.Tensor,
        traj: torch.Tensor | None = None,
        traj_loss_mask: torch.Tensor | None = None,
        alpha: float = 1.0,
        action_dim_loss_mask: torch.Tensor | None = None,
        fast_action_indicator: torch.Tensor | None = None,
        subtask_indicator: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
        num_embodiments: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Computes the total loss.

        Args:
            model_pred (dict[str, torch.Tensor]): The dictionary of model predictions.
            gt_lang_tokens (torch.Tensor): The ground truth language tokens.
            lang_loss_masks (torch.Tensor): The mask for the language loss.
            action_loss_mask (torch.Tensor): The mask for the action diffusion loss.
            traj (torch.Tensor | None, optional): The ground truth trajectory. Defaults to None.
            traj_loss_mask (torch.Tensor | None, optional): The mask for the trajectory loss. Defaults to None.
            alpha (float): A weighting factor for the diffusion loss. Defaults to 1.0.
            action_dim_loss_mask (torch.Tensor | None, optional): Optional
                [B, T, D] mask that restricts flow supervision to valid
                action dimensions.
            fast_action_indicator (torch.Tensor | None, optional): Optional
                mask that identifies FAST action tokens in the language stream.
            subtask_indicator (torch.Tensor | None, optional): Optional mask
                that identifies subtask text tokens in the language stream.

        Returns:
            dict[str, torch.Tensor]: A dictionary of computed losses.
        """

        loss_dict = {}

        if action_dim_loss_mask is None:
            has_action = action_loss_mask.any(dim=-1)
        else:
            has_action = (
                action_dim_loss_mask.to(device=action_loss_mask.device, dtype=torch.bool)
                & action_loss_mask.to(dtype=torch.bool)[:, :, None]
            ).any(dim=(-1, -2))
        diffusion_loss = self._compute_diffusion_loss(
            model_pred['v_t'],
            action_loss_mask,
            action_dim_loss_mask,
            embodiment_id,
        )
        if has_action.any():
            loss_dict['diffusion_loss'] = alpha * diffusion_loss[has_action].mean()
        else:
            loss_dict['diffusion_loss'] = diffusion_loss.sum() * 0.0

        loss_dict.update(
            self._per_embodiment_diffusion_metrics(
                diffusion_loss,
                has_action,
                embodiment_id,
                num_embodiments,
            )
        )

        language_zero = model_pred['v_t'].sum() * 0.0
        if 'lang_token_loss' in model_pred or 'lang_logits' in model_pred:
            if 'lang_token_loss' in model_pred:
                llm_loss_terms = self.llm_loss_terms_from_token_loss(
                    model_pred['lang_token_loss'],
                    lang_loss_masks,
                    fast_action_indicator=fast_action_indicator,
                    subtask_indicator=subtask_indicator,
                )
            else:
                llm_loss_terms = self.llm_loss_terms(
                    model_pred['lang_logits'],
                    gt_lang_tokens,
                    lang_loss_masks,
                    fast_action_indicator=fast_action_indicator,
                    subtask_indicator=subtask_indicator,
                    logits_mask=model_pred.get('lang_logits_mask'),
                )
            shifted_lang_loss_masks = lang_loss_masks[:, 1:].to(device=gt_lang_tokens.device, dtype=torch.bool)
            split_active_mask = torch.zeros_like(shifted_lang_loss_masks, dtype=torch.bool)

            total_active_samples = shifted_lang_loss_masks.any(dim=-1)
            loss_dict['llm_loss'] = self._mean_active_sample_loss(
                llm_loss_terms['total_llm_loss'],
                total_active_samples,
            )

            if fast_action_indicator is not None and 'fast_token_llm_loss' in llm_loss_terms:
                fast_active_mask = shifted_lang_loss_masks & fast_action_indicator[:, 1:].to(
                    device=shifted_lang_loss_masks.device,
                    dtype=torch.bool,
                )
                fast_active_samples = fast_active_mask.any(dim=-1)
                loss_dict['metric/fast_token_llm_loss'] = self._mean_active_sample_loss(
                    llm_loss_terms['fast_token_llm_loss'],
                    fast_active_samples,
                )
                loss_dict.update(
                    self._per_domain_fast_token_metrics(
                        llm_loss_terms['fast_token_llm_loss'],
                        fast_active_samples,
                        embodiment_id,
                    )
                )
                split_active_mask |= fast_active_mask
            else:
                loss_dict['metric/fast_token_llm_loss'] = language_zero

            if subtask_indicator is not None and 'subtask_llm_loss' in llm_loss_terms:
                subtask_active_mask = shifted_lang_loss_masks & subtask_indicator[:, 1:].to(
                    device=shifted_lang_loss_masks.device,
                    dtype=torch.bool,
                )
                subtask_active_samples = subtask_active_mask.any(dim=-1)
                loss_dict['metric/subtask_llm_loss'] = self._mean_active_sample_loss(
                    llm_loss_terms['subtask_llm_loss'],
                    subtask_active_samples,
                )
                split_active_mask |= subtask_active_mask
            else:
                loss_dict['metric/subtask_llm_loss'] = language_zero

            generic_active_samples = (shifted_lang_loss_masks & ~split_active_mask).any(dim=-1)
            loss_dict['metric/generic_llm_loss'] = self._mean_active_sample_loss(
                llm_loss_terms['generic_llm_loss'],
                generic_active_samples,
            )
        else:
            loss_dict['llm_loss'] = language_zero
            loss_dict['metric/fast_token_llm_loss'] = language_zero
            loss_dict['metric/subtask_llm_loss'] = language_zero
            loss_dict['metric/generic_llm_loss'] = language_zero

        if 'traj_pred' in model_pred:
            if traj is not None and traj_loss_mask is not None:
                loss_dict['traj_loss'] = self.traj_loss(model_pred['traj_pred'], traj, traj_loss_mask)
            else:
                loss_dict['traj_loss'] = model_pred['traj_pred'].sum() * 0.0

        return loss_dict

