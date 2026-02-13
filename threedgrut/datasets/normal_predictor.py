import torch, pickle, torchvision
import numpy as np
from PIL import Image


from threedgrut_playground.utils.misc import color_normal


class NormalPredictor:
    model = None
    def __init__(self, generate_normals=False):
        if generate_normals and self.model == None:
            self.model = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)

    def get_intrinsics_matrix_from_dict(self, data):
        fx, fy = data["focal_length"]
        cx, cy = data["principal_point"]

        return torch.tensor([[
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]], dtype=torch.float32, device="cuda")

    def predict_normal(self, image_path, intrinsics, t_to_world, normal_path):
        image = torch.tensor(np.asarray(Image.open(image_path)), device="cuda").unsqueeze(0) / 255.0
        t_to_world = torch.tensor(t_to_world).unsqueeze(0).to("cuda", non_blocking=True)

        with torch.no_grad():
            pred_norm = self.model.infer_tensor(image.permute(0, 3, 1, 2), self.get_intrinsics_matrix_from_dict(intrinsics))
        pred_norm = pred_norm.permute(0, 2, 3, 1)

        t_to_world = t_to_world[0, :3, :3].to("cuda", non_blocking=True)

        pred_norm = pred_norm @ t_to_world.T

        pred_norm_cpu = pred_norm[0].cpu()

        with open(f"{normal_path}.pkl", "wb") as f:
            pickle.dump(pred_norm_cpu, f, pickle.HIGHEST_PROTOCOL)

        pred_image = color_normal(pred_norm)

        torchvision.utils.save_image(
            pred_image.squeeze(0).permute(2, 0, 1), f"{normal_path}.png"
        )
        return pred_norm

    def load_normal(self, normal_path):
        with open(f"{normal_path}.pkl", "rb") as f:
            data = pickle.load(f)
        return data

    def save_anything(self, data):
        with open(f"temp.pkl", "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
