# Model Files Setup

This document explains how to download the required `.pt` model files from Hugging Face and copy them into the `/models` folder of the project.

## Overview

The project relies on pre-trained PyTorch model files (`.pt`) which need to be downloaded separately due to their large size. These files are hosted on Hugging Face's model hub.

## Steps to Download and Setup Models

1. **Locate the Model on Hugging Face**

   Go to the Hugging Face model repository URL provided by the project documentation or maintainer. For example:  
   ```
   https://huggingface.co/username/model-name
   ```

2. **Download the `.pt` Files**

   In the model repository page on Hugging Face, find the required PyTorch model file(s) usually with a `.pt` extension (e.g., `model.pt`, `best_model.pt`). Download these files manually by clicking on the file and selecting the "Download" option.

   Alternatively, for command line download (if the model repo supports direct URL download):
   ```bash
   wget https://huggingface.co/username/model-name/resolve/main/model.pt -P ./models
   ```
   or
   ```bash
   curl -L -o ./models/model.pt https://huggingface.co/username/model-name/resolve/main/model.pt
   ```

3. **Copy Files into `/models` Folder**

   After downloading the `.pt` files, copy or move them to the `/models` directory in your project root folder:
   ```bash
   mkdir -p models
   cp /path/to/downloaded/model.pt models/
   ```

4. **Verify**

   Confirm the `.pt` files exist in the `/models` folder before running the project:
   ```bash
   ls models/
   ```

## Notes

- The `/models` directory is expected to reside at the root of your project.
- Make sure you download the exact model files compatible with the project code.
- If you use any automated scripts or pipelines, adjust the download paths accordingly.
- For large models, consider using Git LFS or huggingface-cli to manage downloads efficiently if supported.

---

Following these steps ensures the models are correctly placed for your application to load and use them.