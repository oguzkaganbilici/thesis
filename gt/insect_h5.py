import h5py

FILES = [
    "trt/trt_feat_visual_inceptionv3.h5",
    "trt/trt_feat_audio_ast.h5",
    "trt/trt_feat_text_berturk.h5",
    "trt/trt_gt.h5",

]

def show(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  [DATASET] {name:40s} shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"  [GROUP]   {name}")
    for k, v in obj.attrs.items():
        print(f"      attr: {k} = {v}")

for path in FILES:
    print(f"\n===== {path} =====")
    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            print(f"  root-attr: {k} = {v}")
        f.visititems(show)