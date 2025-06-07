import torch
import numpy as np
import mcubes
import os
import sys
from datetime import datetime
sys.path.append('.')  # to import from nerf-pytorch

from run_nerf import create_nerf

# === CONFIGURATION ===
model_path = r'D:/Projects/GAI LAb/final eng project/Task2/nerf-pytorch/logs/trex/horns_100_122831_1.0/'
model_name =f'{model_path}020000.tar'

time_str = datetime.now().strftime('%H%M%S')
obj_folder_path = os.path.join(model_path, '3d_obj_{time_str}')
os.makedirs(obj_folder_path, exist_ok=True)
N = 256  # Grid resolution (try 128 if slow)
density_threshold = 30.0  # tweak for shape
output_mesh_path = os.path.join(obj_folder_path,'nerf_mesh.obj')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device=>',device)
# === Load NeRF model ===
_, render_kwargs_test, _, _ = create_nerf()
network_fine = render_kwargs_test['network_fine']

# Load weights
checkpoint = torch.load(model_path, map_location=device)
network_fine.load_state_dict(checkpoint['network_fine_state_dict'])

print(f"[INFO] Loaded model from {model_path}")

# === Sample space ===
print("[INFO] Sampling density grid...")
lin = torch.linspace(-1.5, 1.5, N)
grid = torch.stack(torch.meshgrid(lin, lin, lin), -1).reshape(-1, 3).to(device)

CHUNK = 4096
sigmas = []
network_fine.eval()
with torch.no_grad():
    for i in range(0, grid.shape[0], CHUNK):
        chunk = grid[i:i+CHUNK]
        density = network_fine(chunk)[..., -1]
        sigmas.append(density.cpu())
sigmas = torch.cat(sigmas, dim=0).numpy().reshape(N, N, N)

# === Extract mesh using Marching Cubes ===
print("[INFO] Running Marching Cubes...")
vertices, triangles = mcubes.marching_cubes(sigmas, density_threshold)

# Normalize to original coordinate system
vertices = vertices / float(N) * 3.0 - 1.5

# Save as OBJ
mcubes.export_obj(vertices, triangles, output_mesh_path)
print(f"[DONE] Saved mesh to {output_mesh_path}")
