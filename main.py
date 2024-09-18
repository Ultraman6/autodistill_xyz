import os
import yaml
import requests
import zipfile
from tqdm import tqdm
import supervision as sv
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM


# Step 1: 加载配置文件
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Step 2: 下载并解压视频
def download_and_extract_videos(video_url, video_dir):
    video_zip_path = os.path.join(video_dir, "videos.zip")

    print(f"Downloading videos from {video_url}...")
    response = requests.get(video_url)
    with open(video_zip_path, 'wb') as f:
        f.write(response.content)

    print(f"Extracting videos to {video_dir}...")
    with zipfile.ZipFile(video_zip_path, 'r') as zip_ref:
        zip_ref.extractall(video_dir)

    print("Video download and extraction complete.")


# Step 3: 从视频中提取帧
def extract_frames_from_videos(video_dir, image_dir, frame_stride):
    video_paths = sv.list_files_with_extensions(directory=video_dir, extensions=["mov", "mp4"])

    for video_path in tqdm(video_paths, desc="Extracting frames from videos"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        image_name_pattern = video_name + "-{:05d}.png"

        with sv.ImageSink(target_dir_path=image_dir, image_name_pattern=image_name_pattern) as sink:
            for image in sv.get_video_frames_generator(source_path=video_path, stride=frame_stride):
                sink.save_image(image=image)


# Step 4: 自动标注图片
def auto_label_images(image_dir, dataset_dir, ontology_mapping):
    ontology = CaptionOntology(ontology_mapping)
    base_model = GroundedSAM(ontology=ontology)

    print("Starting auto-labeling of images...")
    base_model.label(input_folder=image_dir, output_folder=dataset_dir)


# Step 5: 处理数据集，区分视频或图片类型
def process_dataset(input_type, config):
    dataset_dir = config['data']['dataset_dir']

    if input_type == "video":
        # 视频类型数据集，下载视频并提取帧
        print("Dataset type is 'video'. Processing video dataset...")

        video_config = config['data']['video']
        video_url = video_config['video_url']
        video_dir = video_config['video_dir']
        frame_stride = video_config['frame_stride']

        # 创建必要目录
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        download_and_extract_videos(video_url, video_dir)
        extract_frames_from_videos(video_dir, dataset_dir, frame_stride)

    elif input_type == "image":
        # 图像类型数据集，假设图像已经存在
        print("Dataset type is 'image'. Processing existing image dataset...")

        image_dir = config['data']['image']['image_dir']
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory {image_dir} does not exist.")

        os.makedirs(dataset_dir, exist_ok=True)

    else:
        raise ValueError("Invalid input type. Must be 'video' or 'image'.")


# Step 6: 主函数，执行自动构造数据集
def main(config_path):
    # 加载配置
    config = load_config(config_path)

    # 提取数据类型
    input_type = config['data']['input_type']

    # 处理数据集（根据视频或图片类型进行处理）
    process_dataset(input_type, config)

    # 提取本体（Ontology）的映射关系
    ontology_mapping = config['ontology']

    # 提取图像目录
    image_dir = config['data']['image']['image_dir']
    dataset_dir = config['data']['dataset_dir']

    # 使用 GroundedSAM 对图片进行自动标注
    auto_label_images(image_dir, dataset_dir, ontology_mapping)

    print("Dataset construction complete.")


# Step 7: 运行脚本
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automate dataset construction for YOLO training.")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to the YAML config file")
    args = parser.parse_args()

    # 执行主程序
    main(args.config)
