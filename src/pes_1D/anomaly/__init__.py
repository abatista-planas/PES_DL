"""Anomaly-detection benchmark for the 1D discriminator, applied to MIT-BIH ECG.

The PES discriminator (``pes_1D.discriminator.CnnDiscriminator``) is, in effect, a
learned 1D-curve shape-anomaly detector. In-domain PES data is too small/unlabeled to
train it, so this package benchmarks the *architecture* on a real, labeled, large
1D-curve dataset with a published GAN comparable: the MIT-BIH arrhythmia database
(BeatGAN, AUROC ~= 0.945, F1 ~= 0.816).
"""
