"""Utilities for ESM models.
"""
import esm_adapter

def get_esm_pretained_model(model_architecture, num_end_adapter_layers):

  if model_architecture == "esm2_t48_15B_UR50D":
    return esm_adapter.pretrained.esm2_t48_15B_UR50D(num_end_adapter_layers)

  if model_architecture == "esm2_t36_3B_UR50D":
    return esm_adapter.pretrained.esm2_t36_3B_UR50D(num_end_adapter_layers)

  if model_architecture == "esm2_t33_650M_UR50D":
    return esm_adapter.pretrained.esm2_t33_650M_UR50D(num_end_adapter_layers)

  if model_architecture == "esm2_t30_150M_UR50D":
    return esm_adapter.pretrained.esm2_t30_150M_UR50D(num_end_adapter_layers)

  if model_architecture == "esm2_t12_35M_UR50D":
    return esm_adapter.pretrained.esm2_t12_35M_UR50D(num_end_adapter_layers)

  if model_architecture == "esm2_t6_8M_UR50D":
    return esm_adapter.pretrained.esm2_t6_8M_UR50D(num_end_adapter_layers)


def load_model(model_architecture, num_end_adapter_layers):
  
  # print("Loading model: ", model_architecture)
  # print("Number of end adapter layers: ", num_end_adapter_layers)

  esm2_model, _ = get_esm_pretained_model(model_architecture,
                                          num_end_adapter_layers)

  return esm2_model