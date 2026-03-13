# Model Sharing Guide

This document outlines the steps to share trained models for the DeepSceneLoc project.

## Recommended Platforms for Sharing

1. **Google Drive**
   - Upload the model files (e.g., `best_model.pth`) to a shared Google Drive folder.
   - Ensure the folder permissions are set to "Anyone with the link can view."
   - Add the download link to the project README.

2. **Hugging Face Hub**
   - Create a repository on [Hugging Face Hub](https://huggingface.co/).
   - Use the `huggingface_hub` Python library to upload the model:
     ```python
     from huggingface_hub import HfApi

     api = HfApi()
     api.upload_file(
         path_or_fileobj="models/checkpoints/best_model.pth",
         path_in_repo="best_model.pth",
         repo_id="<your-username>/<repo-name>",
         repo_type="model"
     )
     ```
   - Add the model card and usage instructions.

3. **Amazon S3**
   - Upload the model to an S3 bucket.
   - Use the AWS CLI to upload:
     ```bash
     aws s3 cp models/checkpoints/best_model.pth s3://<your-bucket-name>/best_model.pth
     ```
   - Ensure the bucket policy allows public read access.
   - Share the S3 URL in the README.

4. **GitHub Releases**
   - Use GitHub Releases to attach the model file to a release.
   - Navigate to the "Releases" section of your GitHub repository.
   - Create a new release and upload the model file.

## Updating Documentation

- Add the model sharing details to the `README.md` under a new section titled "Model Sharing."
- Include links to the shared models and usage instructions.

---

## Example Usage

To load the shared model:
```python
import torch
from models.model import DeepSceneLocResNet50

model = DeepSceneLocResNet50(num_classes=5)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
```

---

For any issues, contact the project team.