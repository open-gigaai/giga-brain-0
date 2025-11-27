import torch
import torch.nn as nn
import torch.nn.functional as F


class GigaBrain0Loss(nn.Module):
    """Loss function for GigaBrain0, including diffusion loss for actions,
    cross-entropy loss for language tokens, and MSE loss for trajectory
    prediction."""

    def __init__(self):
        """Initializes the GigaBrain0Loss module."""
        super().__init__()
        self.x_t: torch.Tensor | None = None
        self.u_t: torch.Tensor | None = None
        self.time: torch.Tensor | None = None

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
        """Samples from a Beta distribution using two Gamma variables.

        Args:
            alpha (float): The alpha parameter of the Beta distribution.
            beta (float): The beta parameter of the Beta distribution.
            bsize (int): The batch size.
            device (torch.device): The device to place the tensors on.

        Returns:
            torch.Tensor: Samples from the Beta distribution.
        """
        gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
        gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
        return gamma1 / (gamma1 + gamma2)

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

    def add_noise(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Adds noise to actions for the diffusion process and stores
        intermediate values.

        Args:
            actions (torch.Tensor): The original actions.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the noisy actions and the timesteps.
        """
        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        self.x_t = x_t
        self.u_t = u_t
        self.time = time

        return x_t, time

    def llm_loss(self, logits: torch.Tensor, gt_lang_tokens: torch.Tensor, lang_loss_masks: torch.Tensor) -> torch.Tensor:
        """Calculates the cross-entropy loss for language model predictions.

        Args:
            logits (torch.Tensor): The predicted logits from the language model.
            gt_lang_tokens (torch.Tensor): The ground truth language tokens.
            lang_loss_masks (torch.Tensor): The mask to apply to the language loss.

        Returns:
            torch.Tensor: The calculated language model loss.
        """
        logits = logits[:, :-1, :]  # remove the last token <eos>

        lang_loss_masks = lang_loss_masks[:, 1:]
        gt_lang_tokens = gt_lang_tokens[:, 1:]

        vocab_size = logits.shape[-1]
        llm_loss = F.cross_entropy(logits.reshape(-1, vocab_size), gt_lang_tokens.reshape(-1).long(), reduction='none')
        llm_loss = llm_loss.reshape(gt_lang_tokens.shape[0], gt_lang_tokens.shape[1]) * lang_loss_masks

        llm_loss = llm_loss.sum(axis=-1) / torch.clamp(lang_loss_masks.sum(axis=-1), min=1)

        return llm_loss

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

    def forward(
        self,
        model_pred: dict[str, torch.Tensor],
        gt_lang_tokens: torch.Tensor,
        lang_loss_masks: torch.Tensor,
        action_loss_mask: torch.Tensor,
        traj: torch.Tensor | None = None,
        traj_loss_mask: torch.Tensor | None = None,
        alpha: float = 1.0,
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

        Returns:
            dict[str, torch.Tensor]: A dictionary of computed losses.
        """

        loss_dict = {}

        diffusion_loss = F.mse_loss(self.u_t, model_pred['v_t'], reduction='none')
        diffusion_loss = diffusion_loss.mean(axis=-1) * action_loss_mask
        diffusion_loss = diffusion_loss.mean(axis=-1)
        loss_dict['diffusion_loss'] = alpha * diffusion_loss

        if 'lang_logits' in model_pred:
            loss_dict['llm_loss'] = self.llm_loss(model_pred['lang_logits'], gt_lang_tokens, lang_loss_masks)

        if 'traj_pred' in model_pred and traj is not None and traj_loss_mask is not None:
            loss_dict['traj_loss'] = self.traj_loss(model_pred['traj_pred'], traj, traj_loss_mask)

        return loss_dict
