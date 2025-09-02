# EEG-to-Image Reconstruction ( MindViz )

This project implements a novel **EEG-to-Image reconstruction** framework using text-guided diffusion.  
Instead of mapping EEG signals directly to pixels, we align EEG-derived embeddings with text prompts.  
An EEG encoder is trained alongside a text retrieval model to output class-level descriptions, which are then fed into a Stable Diffusion model to generate images.  

By leveraging EEG’s high-level semantic cues, our method avoids brittle pixel-level decoding and produces reconstructions that are both more coherent and category-faithful.

---

## 🚀 Methodology of MindViz

### Dual-Backbone Architecture
- **Image Backbone (ViT-H/14):** Maps EEG features into a visual feature embedding.  
- **Text Backbone (ResNet-50):** Predicts class-level text prompts aligned with EEG features.  
- These two pathways capture complementary strengths for EEG representation.

### Text-Guided Generation
- The text prompts predicted from EEG are used to **condition Stable Diffusion**.  
- This allows the diffusion model to generate images that reflect the semantic content of the EEG signals.  

### Novelty
- Previous works treat EEG-to-image as a **denoising problem**.  
- Our method reframes it as a **semantic guidance problem**, where text derived from EEG directs generation, leveraging high retrieval scores in the reconstriction task.  
- This shift leverages EEG’s strength in encoding **abstract semantics** and avoids brittle pixel-level mappings.  

---

## ⚙️ Installation
*(Instructions for setting up the environment and installing dependencies will be added here.)*

---

## 🏋️ Training
*(Detailed training instructions, including data preparation and training commands, will be added here.)*

---

## 🎨 Inference
*(Guidance for running the trained model to generate images from EEG data will be added here.)*

---

## 📊 Results

- Reconstructions are **more coherent and semantically faithful** than baselines.  
- Compared to ATM retrieval and Uncertainty-Aware Blur Prior (UBP), our method recovers more accurate object categories.  

---

## 👥 Contributors
- Abdulaziz Alkhateeb  
- Amjad Alqasemi  
- Husain Althagafi  
- Abdulellah Mojallad  
- Musa Ibn Rashid  

---

## 📜 License
*(License information to be added here.)*
