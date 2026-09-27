import os, sys
import json
import torch
import numpy as np
import h5py
from transformers import AutoTokenizer, AutoModel
from faster_whisper import WhisperModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import maclar, AKTIF_MAC, TEXT_FEAT_PATH, VISUAL_FEAT_PATH

FULL_MATCH_PATH = maclar[AKTIF_MAC]["full"]


def load_whisper():
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

    return model

def transcribe_match(model, video_path):

    segments, _ = model.transcribe(video_path, language="tr",
                                     task="transcribe", vad_filter=True, beam_size=5)

    segments_list = []

    for seg in segments:
        # print("text: ", seg.text, " start: ", seg.start, " end: ", seg.end)
        segment = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        }
        segments_list.append(segment)

    return segments_list

def load_bert():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    bert_model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
    
    bert_model.eval()
    bert_model.to(device)

    return device, tokenizer, bert_model

def compute_pad_vector(tokenizer, model, device):
    
    pad_token_id = torch.tensor([[tokenizer.pad_token_id]])
    pad_attention = torch.tensor([[1]])
    
    pad_token_id = pad_token_id.to(device)
    pad_attention = pad_attention.to(device)
    
    with torch.no_grad():
        pad_vector = model(input_ids=pad_token_id, attention_mask=pad_attention)
        last_hidden = pad_vector.last_hidden_state[0, 0, :]
        pad_vector_ = last_hidden.cpu().numpy()

    return pad_vector_

def encode_segments(segment_list, tokenizer, model, device):

    cls_list = []
    mean_pooling_list = []

    for seg in segment_list:
        tokens = tokenizer(seg["text"], return_tensors="pt",
                       truncation=True, max_length=512)

        
        tokens = tokens.to(device)

        attn_mask = tokens["attention_mask"]
        attn_mask = attn_mask.unsqueeze(-1).float()
        

        with torch.no_grad():
            out = model(**tokens)
            cls = out.last_hidden_state[:, 0, :].cpu()

            total = (out.last_hidden_state * attn_mask).sum(dim=1)
            mean_pooling = (total / attn_mask.sum(dim=1)).cpu()

            cls_list.append(cls)
            mean_pooling_list.append(mean_pooling)

    cls_ = torch.cat(cls_list, dim=0).numpy()
    mean_ = torch.cat(mean_pooling_list, dim=0).numpy()

    return cls_, mean_
    

def broadcasting(feature_matrix, segment_list, pad_vector, N):

    raw_array = np.tile(pad_vector, (N, 1))

    for i, seg in enumerate(segment_list):
        start_sec = int(seg["start"])
        end_sec = int(seg["end"])
        end_sec = min(end_sec, N-1) # kenar sınırı

        if start_sec >= N:
            continue

        raw_array[start_sec: end_sec + 1] = feature_matrix[i]

    return raw_array

    
if __name__ == "__main__":
    # yeni maç gelince bunu ac
    """whisper_model = load_whisper()

    segments_list = transcribe_match(whisper_model, FULL_MATCH_PATH)

    with open(f"transcript_{AKTIF_MAC}.json", "w", encoding="utf-8") as f:
        json.dump(segments_list, f, ensure_ascii=False, indent=4)
    """

    with open(f"transcript_{AKTIF_MAC}.json", "r", encoding="utf-8") as f:
        segments_list = json.load(f)

    device, tokenizer, bert_model = load_bert()

    pad_vector = compute_pad_vector(tokenizer, bert_model, device)

    cls, mean = encode_segments(segments_list, tokenizer, bert_model, device)

    with h5py.File(VISUAL_FEAT_PATH, "r") as f:
        N = f[AKTIF_MAC]["features"].shape[0]

    cls_feature = broadcasting(cls, segments_list, pad_vector, N)
    mean_feature = broadcasting(mean, segments_list, pad_vector, N)


    np.save(f"feats_text_cls_{AKTIF_MAC}.npy", cls)      # ham [1806, 768]
    np.save(f"feats_text_mean_{AKTIF_MAC}.npy", mean)    # ham [1806, 768]

    
    with h5py.File(TEXT_FEAT_PATH, "a") as f:
        if AKTIF_MAC in f: del f[AKTIF_MAC]
        group = f.create_group(AKTIF_MAC)
        group.create_dataset("features", data=cls_feature)

    # geri okumak icin
    with h5py.File(TEXT_FEAT_PATH, "r") as f:
        print(f[AKTIF_MAC]["features"].shape)   # (5917, 768)
        print(list(f.keys()))                    # ['fcsb-fenerbahce']

