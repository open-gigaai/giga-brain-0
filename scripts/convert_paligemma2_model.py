import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from giga_models import GigaBrain07Policy


def convert_vision_tower_weights(old_state_dict, new_state_dict, old_prefix, new_prefix):
    """Convert SiglipVisionTransformer weights."""
    # This function is identical to the one for paligemma, as the vision tower structure is the same.
    keys = list(old_state_dict.keys())  # Convert to list to avoid RuntimeError during iteration
    for key in keys:
        if key.startswith(old_prefix):
            new_key = new_prefix + key[len(old_prefix) :]
            new_state_dict[new_key] = old_state_dict[key]
            old_state_dict.pop(key)


def convert_projector_weights(old_state_dict, new_state_dict, old_prefix, new_prefix):
    """Convert PaliGemmaMultiModalProjector weights."""
    for param in ['weight', 'bias']:
        old_key = f'{old_prefix}.linear.{param}'
        new_key = f'{new_prefix}.linear.{param}'
        if old_key in old_state_dict:
            new_state_dict[new_key] = old_state_dict[old_key]
            old_state_dict.pop(old_key)


def convert_embedding_weights(old_state_dict, new_state_dict, old_prefix, new_prefix):
    """Convert embedding layer weights."""
    old_key = f'{old_prefix}.embed_tokens.weight'
    new_key = f'{new_prefix}.embed_tokens.weight'
    if old_key in old_state_dict:
        new_state_dict[new_key] = old_state_dict[old_key]
        old_state_dict.pop(old_key)


def convert_decoder_layer_weights(old_state_dict, new_state_dict, layer_idx):
    new_layer_prefix = f'paligemma_with_expert.layers.{layer_idx}'

    # PaliGemma2 part (index 0)
    pali_old_prefix = f'language_model.model.layers.{layer_idx}'

    # Attention projections
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        old_key = f'{pali_old_prefix}.self_attn.{proj}.weight'
        new_key = f'{new_layer_prefix}.self_attn.{proj}.0.weight'
        if old_key in old_state_dict:
            new_state_dict[new_key] = old_state_dict[old_key]
            old_state_dict.pop(old_key)

    # MLP
    for proj in ['gate_proj', 'up_proj', 'down_proj']:
        old_key = f'{pali_old_prefix}.mlp.{proj}.weight'
        new_key = f'{new_layer_prefix}.mlps.0.{proj}.weight'
        if old_key in old_state_dict:
            new_state_dict[new_key] = old_state_dict[old_key]
            old_state_dict.pop(old_key)

    # Layer norms for Gemma2 (4 norms)
    for norm_name, new_norm_name in [
        ('input_layernorm', 'input_layernorms.0'),
        ('post_attention_layernorm', 'post_attention_layernorms.0'),
        ('pre_feedforward_layernorm', 'pre_feedforward_layernorms.0'),
        ('post_feedforward_layernorm', 'post_feedforward_layernorms.0'),
    ]:
        old_key = f'{pali_old_prefix}.{norm_name}.weight'
        new_key = f'{new_layer_prefix}.{new_norm_name}.weight'
        if old_key in old_state_dict:
            new_state_dict[new_key] = old_state_dict[old_key]
            old_state_dict.pop(old_key)


def convert_final_norm_weights(old_state_dict, new_state_dict):
    # PaliGemma2 norm (index 0)
    old_key = 'language_model.model.norm.weight'
    new_key = 'paligemma_with_expert.norms.0.weight'
    if old_key in old_state_dict:
        new_state_dict[new_key] = old_state_dict[old_key]
        old_state_dict.pop(old_key)


def convert_v1_to_v2_state_dict(old_state_dict, num_hidden_layers=26):
    """Convert state_dict."""
    new_state_dict = {}

    print('Converting Vision Tower weights...')
    convert_vision_tower_weights(old_state_dict, new_state_dict, 'vision_tower.vision_model', 'paligemma_with_expert.vision_tower')

    print('Converting Multi-Modal Projector weights...')
    convert_projector_weights(old_state_dict, new_state_dict, 'multi_modal_projector', 'paligemma_with_expert.multi_modal_projector')

    print('Converting Embedding weights...')
    convert_embedding_weights(old_state_dict, new_state_dict, 'language_model.model', 'paligemma_with_expert')

    print(f'Converting {num_hidden_layers} Decoder Layer weights...')
    for i in range(num_hidden_layers):
        if (i + 1) % 5 == 0:
            print(f'  Processing layer {i+1}/{num_hidden_layers}')
        convert_decoder_layer_weights(old_state_dict, new_state_dict, i)

    print('Converting Final Norm weights...')
    convert_final_norm_weights(old_state_dict, new_state_dict)

    return new_state_dict


def load_v1_model(input_path: str):
    """Load the state_dict of a v1 model."""
    input_path = Path(input_path)

    safetensors_files = list(input_path.glob('*.safetensors'))
    if safetensors_files:
        print(f'Loading model from safetensors files: {safetensors_files}')
        state_dict = {}
        for sf_file in safetensors_files:
            state_dict.update(load_file(str(sf_file), device='cpu'))
        return state_dict

    raise FileNotFoundError(f'Model files not found in {input_path} (expected *.safetensors or diffusion_pytorch_model*.bin)')


def save_v2_model(new_state_dict, output_path: str, num_embodiments: int):
    """Save the v2 model."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print('Creating v2 GigaBrain07Policy (PaliGemma2) model instance...')
    model = GigaBrain07Policy(
        num_embodiments=num_embodiments,
        enable_learnable_traj_token=False,
        enable_knowledge_insulation=True,
        vision_in_channels=3,
        vlm_type='paligemma2',  # IMPORTANT: Use PaliGemma2 architecture
        vlm_hidden_size=2304,
    )

    # Load the converted weights
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    if missing_keys:
        print(f'Missing keys ({len(missing_keys)}): {missing_keys}...')

    assert len(unexpected_keys) == 0, f'Unexpected keys: {unexpected_keys}'

    # Save new_state_dict directly
    state_dict_path = output_path / 'pytorch_model.bin'
    print(f'Saving state_dict directly to {state_dict_path}...')
    torch.save(new_state_dict, str(state_dict_path))
    print(f'State dict saved to {state_dict_path}')

    # shared tensors can not be saved in safetensors, so we need to un-tie the lm_head and embed_tokens
    model.paligemma_with_expert.lm_head.weight = torch.nn.Parameter(model.paligemma_with_expert.lm_head.weight.detach().clone())
    # Save the model
    print(f'Saving v2 model to {output_path}...')
    model.save_pretrained(str(output_path), safe_serialization=True)
    print(f'Model saved to {output_path}')

    # Verify that the model can be loaded
    print('Verifying model loading...')
    del model
    model_loaded = GigaBrain07Policy.from_pretrained(str(output_path))
    print('Model verification successful')
    del model_loaded


def main():
    parser = argparse.ArgumentParser(description='Convert PaliGemma2 model')
    parser.add_argument('--input_path', type=str, required=True, help='Input model path')
    parser.add_argument('--output_path', type=str, required=True, help='Output model path')
    parser.add_argument('--num_embodiments', type=int, default=2, help='Number of embodiments')
    parser.add_argument('--num_hidden_layers', type=int, default=26, help='Number of Gemma2 hidden layers')

    args = parser.parse_args()

    print('=' * 80)
    print('PaliGemma2 Model Conversion:')
    print('=' * 80)
    print(f'Input path: {args.input_path}')
    print(f'Output path: {args.output_path}')
    print(f'Embodiments: {args.num_embodiments}')
    print(f'Hidden Layers: {args.num_hidden_layers}')
    print('=' * 80)

    # Load v1 model
    print('\nStep 1: Loading v1 model...')
    old_state_dict = load_v1_model(args.input_path)
    print(f'Loaded {len(old_state_dict)} weights')

    # Convert state_dict
    print('\nStep 2: Converting state_dict...')
    new_state_dict = convert_v1_to_v2_state_dict(
        old_state_dict,
        num_hidden_layers=args.num_hidden_layers,
    )
    print(f'Conversion complete, new model has {len(new_state_dict)} weights')

    # Save v2 model
    print('\nStep 3: Saving v2 model...')
    save_v2_model(new_state_dict, args.output_path, args.num_embodiments)

    print('\n' + '=' * 80)
    print('Conversion complete!')
    print('=' * 80)


if __name__ == '__main__':
    main()
