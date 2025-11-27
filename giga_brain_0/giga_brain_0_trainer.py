from typing import Any

import torch
from giga_models import GigaBrain0Policy
from giga_train import Trainer

from .giga_brain_0_loss import GigaBrain0Loss


class GigaBrain0Trainer(Trainer):
    def get_models(self, model_config: dict[str, Any]) -> GigaBrain0Policy:
        """Initializes and returns the GigaBrain0Policy model.

        Args:
            model_config (dict[str, Any]): Configuration dictionary for the model.

        Returns:
            GigaBrain0Policy: The initialized GigaBrain0Policy model.
        """
        if hasattr(model_config, 'pretrained'):
            pretrained = model_config.pop('pretrained')
            giga_brain_0 = GigaBrain0Policy.from_pretrained(pretrained)
            if len(model_config.keys()) > 0:
                giga_brain_0 = process_model(giga_brain_0, model_config)
        else:
            giga_brain_0 = GigaBrain0Policy(**model_config)
            if hasattr(model_config, 'pretrained_paligemma_path'):
                pretrained_paligemma_state_dict = torch.load(model_config.pretrained_paligemma_path, map_location='cpu')

                weight = pretrained_paligemma_state_dict['paligemma_with_expert.vision_tower.embeddings.patch_embedding.weight']
                new_weight = _resize_patch_embedding_weight(weight, giga_brain_0.vision_in_channels)
                pretrained_paligemma_state_dict['paligemma_with_expert.vision_tower.embeddings.patch_embedding.weight'] = new_weight

                _, unexpected_keys = giga_brain_0.load_state_dict(pretrained_paligemma_state_dict, strict=False)
                if unexpected_keys:
                    raise ValueError(f'Unexpected keys: {unexpected_keys}')

        if giga_brain_0.enable_next_token_prediction:
            # Make sure the lm_head and embed_tokens are tied
            assert giga_brain_0.paligemma_with_expert.lm_head.weight.data_ptr() == giga_brain_0.paligemma_with_expert.embed_tokens.weight.data_ptr()

        giga_brain_0.to(self.device)
        giga_brain_0.train()

        self.loss_func = GigaBrain0Loss()

        return giga_brain_0

    def forward_step(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Performs a forward pass and calculates the loss.

        Args:
            batch_dict (dict[str, torch.Tensor]): A dictionary containing the batch data.

        Returns:
            dict[str, torch.Tensor]: A dictionary of losses.
        """
        images = batch_dict['images']
        img_masks = batch_dict['image_masks']
        lang_tokens = batch_dict['lang_tokens']
        lang_masks = batch_dict['lang_masks']

        lang_att_masks = batch_dict['lang_att_masks']
        lang_loss_masks = batch_dict['lang_loss_masks']
        fast_action_indicator = batch_dict['fast_action_indicator']

        actions = batch_dict['action']
        action_loss_mask = batch_dict['action_loss_mask']

        traj = None
        traj_loss_mask = None
        if 'traj' in batch_dict:
            traj = batch_dict['traj']
            traj_loss_mask = batch_dict['traj_loss_mask']

        emb_ids = batch_dict['embodiment_id']

        noisy_model_input, timesteps = self.loss_func.add_noise(actions)
        model_pred = self.model(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            noisy_model_input,
            timesteps,
            emb_ids,
            lang_att_masks=lang_att_masks,
            fast_action_indicator=fast_action_indicator,
        )

        loss = self.loss_func(model_pred, lang_tokens, lang_loss_masks, action_loss_mask, traj, traj_loss_mask)
        return loss


def process_model(model: GigaBrain0Policy, model_config: dict[str, Any]) -> GigaBrain0Policy:
    """Processes a pre-trained model with an updated configuration. This is
    mainly used for resizing the patch embedding layer.

    Args:
        model (GigaBrain0Policy): The pre-trained GigaBrain0Policy model.
        model_config (dict[str, Any]): The updated model configuration.

    Returns:
        GigaBrain0Policy: The processed model with updated weights.
    """
    state_dict = model.state_dict()
    weight = state_dict['paligemma_with_expert.vision_tower.embeddings.patch_embedding.weight']

    updated_model_config = dict(model.config)
    for key in model_config:
        if key in updated_model_config:
            updated_model_config[key] = model_config[key]

    new_model = GigaBrain0Policy(**updated_model_config)

    new_weight = _resize_patch_embedding_weight(weight, updated_model_config['vision_in_channels'])
    state_dict['paligemma_with_expert.vision_tower.embeddings.patch_embedding.weight'] = new_weight
    new_model.load_state_dict(state_dict, strict=False)
    del model
    model = new_model

    return model


def _resize_patch_embedding_weight(weight: torch.Tensor, target_in_channels: int) -> torch.Tensor:
    """Resizes the patch embedding weights to match the target number of input
    channels.

    Args:
        weight (torch.Tensor): The original patch embedding weights.
        target_in_channels (int): The target number of input channels.

    Returns:
        torch.Tensor: The resized patch embedding weights.
    """
    current_in_channels = weight.shape[1]
    if current_in_channels == target_in_channels:
        return weight
    if current_in_channels > target_in_channels:
        return weight[:, :target_in_channels, :, :]
    new_shape = list(weight.shape)
    new_shape[1] = target_in_channels
    new_weight = weight.new_zeros(new_shape)
    new_weight[:, :current_in_channels, :, :] = weight
    return new_weight
