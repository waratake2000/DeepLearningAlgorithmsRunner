import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_PATH = "/root/result"
DATASET_PATH = "/root/dataset/Annotated_High-Resolution_Anime"
ANNOTATION_DATA = "/root/dataset/Annotated_High-Resolution_Anime/annotations/annotations_1_500.csv"
TEST_SPLIT = 0.05
RESIZE = 224
