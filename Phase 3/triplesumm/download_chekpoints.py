from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="smkim37/TripleSumm", 
                       filename="best_model_ckpt_mrhisum.pth",
                       local_dir="./checkpoint")

print(path)