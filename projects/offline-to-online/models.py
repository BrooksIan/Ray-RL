"""Shared BC / PPO RLModules so BC weights are a strict subset of PPO.

Pattern follows Ray RLlib's offline BC→PPO example: train a policy-only BC
module, then load those weights into a PPO module that adds a value head.
"""

from __future__ import annotations

from torch import nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.models.configs import MLPEncoderConfig, MLPHeadConfig
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override

# CartPole-v1 and TicketQueue-v0 both use obs dim 4 and Discrete(2).
OBS_DIM = 4
ACTION_DIM = 2
HIDDEN = 128


class BCPolicyModule(TorchRLModule):
    """Policy-only module for offline BC."""

    @override(TorchRLModule)
    def setup(self):
        self._encoder = MLPEncoderConfig(
            input_dims=[OBS_DIM],
            hidden_layer_dims=[HIDDEN, HIDDEN],
            hidden_layer_activation="relu",
            output_layer_dim=None,
        ).build(framework="torch")
        self._pi = MLPHeadConfig(
            input_dims=[HIDDEN],
            hidden_layer_dims=[HIDDEN],
            hidden_layer_activation="relu",
            output_layer_dim=ACTION_DIM,
            output_layer_activation="linear",
        ).build(framework="torch")

    @override(TorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        return {
            Columns.ACTION_DIST_INPUTS: self._pi(self._encoder(batch)[ENCODER_OUT])
        }

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward_inference(batch)


class PPOPolicyModule(BCPolicyModule, ValueFunctionAPI):
    """PPO module: BC policy trunk + zero-init value head (avoids a hard jolt)."""

    @override(BCPolicyModule)
    def setup(self):
        super().setup()
        self._vf = MLPHeadConfig(
            input_dims=[HIDDEN],
            hidden_layer_dims=[HIDDEN],
            hidden_layer_activation="relu",
            hidden_layer_weights_initializer=nn.init.zeros_,
            hidden_layer_bias_initializer=nn.init.zeros_,
            output_layer_dim=1,
            output_layer_activation="linear",
            output_layer_weights_initializer=nn.init.zeros_,
            output_layer_bias_initializer=nn.init.zeros_,
        ).build(framework="torch")

    @override(BCPolicyModule)
    def _forward_train(self, batch, **kwargs):
        features = self._encoder(batch)[ENCODER_OUT]
        return {
            Columns.ACTION_DIST_INPUTS: self._pi(features),
            Columns.VF_PREDS: self._vf(features).squeeze(-1),
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings = self._encoder(batch)[ENCODER_OUT]
        return self._vf(embeddings).squeeze(-1)
