# pip install image-reward
import ImageReward as RM
import torch
from pathlib import Path
from tqdm import tqdm
import shutil

model = RM.load("ImageReward-v1.0")
prompt = "Too long chin"
prompt = "square-jawed"
prompt2 = prompt.replace(" ", "_")

def single_run(prompt: str, img_path: str) -> float:
    with torch.no_grad():
        reward = model.score(prompt, [img_path])
    return reward

inn = Path("/mnt/d/v1/").glob("v1_50*.png")

Path(f"/mnt/d/v1{prompt2}").mkdir(exist_ok=True)
for i in tqdm(list(inn)):
    reward = single_run(prompt, str(i))
    reward = reward+1
    print(reward, str(i))
    dstpath = f"/mnt/d/v1{prompt2}/{reward}_{i.stem}"
    shutil.copy(i , dstpath)
