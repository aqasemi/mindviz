from huggingface_hub import snapshot_download, upload_file

upload_file(
    path_or_fileobj="/root/mjo/datasets/things-eeg/Preprocessed_data_250Hz_whiten/sub-08/test.pt",
    path_in_repo="preprocessed_data/sub-08/test.pt",
    repo_id="xmjo/jkk",
    repo_type="model"
)
# snapshot_download(
#     repo_id="LidongYang/EEG_Image_decode",
#     repo_type="dataset",
#     local_dir="/root/mjo/datasets/things-eeg/embeddings/",
#     allow_patterns=["ViT-H-14_features_train.pt", "ViT-H-14_features_test.pt"]
# )

# wget -O training.zip https://osf.io/download/3v527/ - train images
# wget -O test.zip https://osf.io/download/znu7b/ - test images
# unzip test.zip -d /root/mjo/datasets/things-eeg/image_set/
# unzip training.zip -d /root/mjo/datasets/things-eeg/image_set/


# wget -O sub-08.zip https://osf.io/download/xr4f9/ - eeg raw data
# unzip sub-08.zip -d /root/mjo/datasets/things-eeg/