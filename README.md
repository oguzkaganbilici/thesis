# Multimodal Video Summarization for Turkish Broadcast Content

Master's thesis project: a **trimodal (visual + audio + text) extractive video summarization system** for Turkish broadcast media (news, sports, documentaries), developed with archival footage from TRT (Turkish Radio and Television Corporation).

## Overview

The system scores every frame of a broadcast video by importance and selects key scenes to form a video summary. It builds on two recent architectures:

- **TripleSumm** (ICLR 2026) — trimodal fusion base: dynamic per-frame modality weighting over visual, audio, and text channels
- **A2Summ** (CVPR 2023) — source of alignment-guided attention masking and contrastive training components

The planned primary contribution is a **hybrid architecture**: TripleSumm's trimodal fusion backbone augmented with A2Summ's segment-aware attention mask and intra-sample contrastive loss, adapted to Turkish via a text-encoder swap (RoBERTa → BERTurk). A RAG layer (natural-language querying over extracted key scenes) is planned as a secondary demonstration.

## Design principles

- **Extractive by design.** Broadcast/news content requires hallucination-free summaries; the system selects existing frames and sentences rather than generating new content. ASR (Whisper) is used only as a preprocessing/modality-conversion step.
- **Frozen encoders, cached features.** All encoders are frozen; features are extracted once and cached to HDF5. Training then touches only the lightweight fusion model (~2.7M parameters), which fits comfortably on a free Colab T4 GPU.
- **Language-independent channels stay untouched.** Only the text encoder changes for Turkish adaptation; visual and audio encoders are language-agnostic.

## Repository structure

| Phase | Contents | Status |
|---|---|---|
| `phase-2-feature-extraction/` | Trimodal feature extraction pipeline (video → HDF5) | ✅ prototype complete (single test video) |
| `phase-3-model/` | Baselines, TripleSumm adaptation, hybrid model | planned |
| `phase-4-rag/` | RAG demo layer over extracted scenes | planned |

## Tech stack

Python · PyTorch · torchaudio · HuggingFace Transformers · OpenAI Whisper · FFmpeg · h5py · Google Colab (T4)

**Models:** InceptionV3 (visual) · AST (audio) · Whisper large (ASR) · BERTurk (text)

## Data

Feature files (`.h5`), videos, and audio are **not** tracked in this repository — they are reproducible from the pipeline code. The target dataset (TRT broadcast videos with editor-annotated key scenes) is institutional and will not be published here.

## References

- TripleSumm: *Triple-Modal Video Summarization* (ICLR 2026)
- A2Summ: He et al., *Align and Attend: Multimodal Summarization with Dual Contrastive Losses* (CVPR 2023)
- BERTurk: Schweter, *BERTurk — BERT models for Turkish* (2020)
- Whisper: Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (2022)
