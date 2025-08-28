import os
import cv2
import torch
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def main():
    cfg = get_cfg()
    cfg.merge_from_file("configs/city_resnet50.yaml")
    cfg.MODEL.WEIGHTS = "./resnet_24000_city/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)

    test_dir = "./foggy_cityscape/VOC2007/JPEGImages"
    output_dir = "./inference_results3"
    os.makedirs(output_dir, exist_ok=True)

    metadata = MetadataCatalog.get("cityscapes_val")

    with open("./foggy_cityscape/VOC2007/ImageSets/Main/test.txt", "r") as f:
        test_files = [line.strip() for line in f.readlines()]

    print(f"共讀取 {len(test_files)} 張測試圖片。")

    for idx,img_id in enumerate(tqdm(test_files, desc="推論中")):
        img_path = os.path.join(test_dir, img_id + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(test_dir, img_id + ".png")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        save_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])

if __name__ == "__main__":
    main()
