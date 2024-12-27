import torch
import ray.cloudpickle as pickle
import pickle as pk

# Small file to save the model based on a checkpoint (if hp tuning failed before returning best model)
data_path = "results/best_checkpoint/checkpoint_000099/data.pkl"
with open(data_path, "rb") as fp:
    best_checkpoint_data = pickle.load(fp)
    best_trained_model = best_checkpoint_data["model"]

transf_path = "results/best_checkpoint/checkpoint_000099/transformation_object.pkl"
with open(transf_path, "rb") as f:
    transformation_object = pk.load(f)

print(best_trained_model)
print(best_trained_model.features_per_model)

model_path = "best_model/best_model"
with open(model_path, "wb") as f:
    torch.save(best_trained_model, f)
transformation_object_path = "best_model/transformation_object.pkl"
with open(transformation_object_path, "wb") as f:
    pk.dump(transformation_object, f)
