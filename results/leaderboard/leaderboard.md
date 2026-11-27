| Rank | Statement | Model | Test peak F1 (3 px) | Peak P | Peak R | Pixel IoU | Test images |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `best_model` | segmentation-models-pytorch Unet with EfficientNet-B4 encoder (ImageNet, smp-hub), full-res decoder, 1-ch input, pointwise log norm, focal x10 + dice; best val checkpoint (step 10000) of h1 run smp_effb4; 4-flip TTA; threshold 0.9995 picked on 300 val images | 0.9217 | 0.8961 | 0.9489 | 0.4947 | 1529 |
| 2 | `hf_upernet` | UperNet (HF transformers) + ConvNeXt-tiny backbone, ADE20k-pretrained, 1-ch log1p/std adapter, PixelShuffle(4) full-res head, focal+dice | 0.8886 | 0.8631 | 0.9157 | 0.4944 | 1529 |
| 3 | `lucid_unet` | lucidrains x-unet XUnet 2D (dim 32, mults [1, 2, 4, 8], attn at 1/8 + mid), 1-ch asinh adapter, full-res head, focal_dice loss | 0.8876 | 0.8972 | 0.8782 | 0.5186 | 1529 |
| 4 | `lucid_segformer` | lucidrains segformer-pytorch Segformer (MiT-b0-like dims 32/64/160/256, from scratch), 1-ch input, saturated-pixel zeroing + log1p/per-image standardisation, PixelShuffle(4) full-res head + residual conv refine, BCE(pos_weight 10)+dice | 0.8255 | 0.7940 | 0.8597 | 0.4100 | 1529 |
| 5 | `peaknet_baseline` | PeakNet (carbonscott/peaknet ConvNeXtV2-tiny-like [96..768]x[3,3,9,3] + BiFPN 256 + seg head), from scratch, 1 binary logit, log1p+std norm, saturated pixels zeroed | 0.8054 | 0.8268 | 0.7851 | 0.2233 | 1529 |
