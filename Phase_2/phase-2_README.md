# Phase 2 — Trimodal Feature Extraction Pipeline

End-to-end feature extraction from a broadcast video into a single HDF5 file, covering all three modalities. Encoders are frozen and used purely as feature extractors; the resulting cache is what the fusion model (Phase 3) will train on.

## Pipeline

```
video.mp4
├── VISUAL   FFmpeg (fps=1) → JPG frames → InceptionV3 (frozen, GAP output)   → [N, 2048]
├── AUDIO    FFmpeg (16 kHz mono WAV) → 1 s slices (+zero-pad) → AST (frozen,
│            mean-pooled over patch tokens)                                    → [N, 768]
└── TEXT     Whisper large (Turkish ASR, timestamped segments)
             → BERTurk (frozen, masked mean pooling)                           → [M, 768]
                                                                                 + [M, 2] timestamps
                                                                                 + M raw sentences
```

All outputs are written to one HDF5 file:

```
trt_feature.h5
└── video_test/
    ├── visual           [N, 2048]   float32
    ├── audio            [N, 768]    float32
    ├── text             [M, 768]    float32
    ├── text_timestamps  [M, 2]      (t_start, t_end) per sentence
    └── text_sentences   [M]         UTF-8 variable-length strings
```

`N` = video duration in seconds (1 fps sampling ⇒ frame index = second).
`M` = number of ASR sentences; `M ≠ N` by design — text is event-based, not
time-uniform. Both frame-axis broadcasting (TripleSumm-style) and segment-mask
alignment (A2Summ-style) can be derived from this raw format, which is why raw
embeddings + timestamps are stored instead of a pre-aligned matrix.

## Key design decisions

- **1 fps sampling** — consecutive frames are near-duplicates; 1 fps keeps sequences short and makes `frame index = second`, which later feeds the A2Summ-style alignment mask directly.
- **Frozen encoders + HDF5 caching** — features are computed once; training reads only the cache (10–100× faster, fits free Colab T4). HDF5 over `.pt` for partial reads and hierarchical layout (`video_id/modality`), matching what TripleSumm/A2Summ data loaders expect.
- **Whisper large as ASR recipe** — on a real news clip with jingle music, `small` produced garbage, `medium` fixed fluency but made systematic proper-noun errors ("Dato" for NATO), `large` resolved both. ASR errors cluster in acoustically hard regions and on proper nouns — i.e., the semantically most valuable words — which motivates the planned noise-robustness experiments.
- **Masked mean pooling for sentences** — `[PAD]` positions carry non-zero hidden states; naive `mean(dim=1)` dilutes short sentences. Pooling = `(H · mask).sum(1) / mask.sum(1)`. Mean over CLS because frozen BERT's CLS (NSP-trained) is a weak sentence representation without fine-tuning (cf. Sentence-BERT).
- **Raw over derived** — sentence embeddings are stored per-sentence with timestamps; any aligned/broadcast variant is derivable in a few lines, the reverse is impossible. Raw sentence text is kept for the RAG demo and qualitative analysis.

## Known open items

1. **Visual dimension mismatch** — torchvision InceptionV3 GAP output is 2048-d, while the TripleSumm checkpoint expects 1024-d visual features. The extraction recipe must be aligned with the official TripleSumm script before the pretrained checkpoint can be used.
2. **Pooling recipes (AST, BERTurk)** — mean pooling chosen here; to be cross-checked against the reference implementation.
3. **ASR refinements** — `initial_prompt` for domain vocabulary, WhisperX for word-level timestamps, punctuation-based sentence splitting (Whisper segments ≈ sentences is a pragmatic assumption).

## Requirements

```
torch, torchvision, torchaudio
transformers
openai-whisper
h5py
ffmpeg (system)
```

## Notes

- The notebook was developed and run on Google Colab (free T4 GPU).
- Data files (`.h5`, `.mp4`, `.wav`, extracted frames) are excluded via `.gitignore` — everything is reproducible from the notebook.
