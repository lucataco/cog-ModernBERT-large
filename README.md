# answerdotai/ModernBERT-large Cog model

This is an implementation of [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).

Download the weights with the command:

    huggingface-cli download answerdotai/ModernBERT-large --local-dir checkpoints

## Basic Usage

To run a safe user prediction:

    cog predict -i prompt="Replicate lets you run AI with an API. Run and fine-tune open-source [MASK]"

Output

    models