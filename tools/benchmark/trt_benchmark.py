"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import tensorrt as trt
import pycuda.driver as cuda
from utils import TimeProfiler
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torchvision

from collections import namedtuple, OrderedDict
import glob
import argparse
from dataset import Dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser Example')
    parser.add_argument('--COCO_dir',
                        dest='infer_dir',
                        type=str,
                        default='/data/COCO2017/val2017',
                        help="Directory for images to perform inference on.")
    parser.add_argument("--engine_dir",
                        dest='models_dir',
                        type=str,
                        help="Directory containing model engine files.")
    parser.add_argument('--busy',
                        action='store_true',
                        help="Flag to indicate that other processes may be running.")
    args = parser.parse_args()
    return args

# 完全端到端，无须后处理
def full_e2e(outputs, *args, **kwargs):
    return outputs

# yolo26n-seg、vajra-v1-nano-seg
def yolo_seg_e2e_postprocess(outputs, img_size=640, *args, **kwargs):
    output0 = outputs['output0'] # [1, 300, 38]
    output1 = outputs['output1'] # [1, 32, 160, 160]
    pred = output0[0]       # [300, 38]
    protos = output1[0]     # [32, 160, 160]
    device = pred.device
    
    # 拆解 300 个候选
    boxes = pred[:, :4]           # [300, 4]  xyxy 绝对坐标
    obj_conf = pred[:, 4]         # [300]
    cls_scores = pred[:, 5]       # [300]
    mask_coeffs = pred[:, 6:]     # [300, 32]
    
    scores = obj_conf * cls_scores        # [300]
    labels = torch.zeros(300, dtype=torch.int64, device=device)  # 单类
    
    # 掩码解码
    c, h, w = protos.shape
    protos_flat = protos.view(c, -1)      # [32, 25600]
    
    # [300, 32] @ [32, 25600] -> [300, 25600] -> [300, 160, 160]
    masks = torch.matmul(mask_coeffs, protos_flat).view(-1, h, w)
    
    # 上采样到 640x640
    masks = masks.unsqueeze(1)          # [300, 1, 160, 160]
    masks = F.interpolate(masks, size=(img_size, img_size),
                          mode='bilinear', align_corners=False) > 0.0
    masks = masks.squeeze(1)            # [300, 640, 640]
    
    return (masks.unsqueeze(0),     # [1, 300, 640, 640]
            boxes.unsqueeze(0),     # [1, 300, 4]
            scores.unsqueeze(0),    # [1, 300]
            labels.unsqueeze(0))    # [1, 300]

# yolo11n-seg、yolov12n-seg
def yolo_seg_nms_postprocess(outputs, img_size=640, conf_thresh=0.001,
                                  iou_thresh=0.7, max_det=300, *args, **kwargs):
    """
    后处理顺序: Conf Filter → NMS → Top-K

    Args:
        outputs:
            output0: [1, 116, 8400]  (xywh(4) + cls(80) + mask_coeff(32))
            output1: [1, 32, 160, 160] prototype masks
        conf_thresh: 置信度阈值，默认 0.001
        iou_thresh: NMS IoU 阈值，默认 0.7
        max_det: 最终最大输出数量，默认 300

    Returns:
        masks:  [1, max_det, 640, 640]
        boxes:  [1, max_det, 4]      xyxy 绝对坐标
        scores: [1, max_det]
        labels: [1, max_det]         int64
    """
    output0 = outputs['output0'] # [1, 116, 8400]
    output1 = outputs['output1'] # [1, 32, 160, 160]
    pred = output0[0].permute(1, 0)     # [8400, 116]
    protos = output1[0]                 # [32, 160, 160]
    device = pred.device

    # 拆解预测头
    boxes_xywh = pred[:, :4]             # [8400, 4]
    cls_scores = pred[:, 4:84]           # [8400, 80]
    mask_coeffs = pred[:, 84:]           # [8400, 32]

    scores, labels = cls_scores.max(dim=1)   # [8400]

    # 置信度过滤
    keep = scores > conf_thresh
    num_candidates = keep.sum().item()

    if num_candidates == 0:
        z_masks = torch.zeros((1, max_det, img_size, img_size), dtype=torch.bool, device=device)
        z_boxes = torch.zeros((1, max_det, 4), dtype=torch.float32, device=device)
        z_scores = torch.zeros((1, max_det), dtype=torch.float32, device=device)
        z_labels = torch.zeros((1, max_det), dtype=torch.int64, device=device)
        return z_masks, z_boxes, z_scores, z_labels

    boxes_f = boxes_xywh[keep]
    scores_f = scores[keep]
    labels_f = labels[keep]
    coeffs_f = mask_coeffs[keep]

    # xywh -> xyxy 绝对坐标
    x, y, w, h = boxes_f[:, 0], boxes_f[:, 1], boxes_f[:, 2], boxes_f[:, 3]
    x1 = (x - w / 2) * img_size
    y1 = (y - h / 2) * img_size
    x2 = (x + w / 2) * img_size
    y2 = (y + h / 2) * img_size
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    # NMS
    nms_idx = torchvision.ops.nms(boxes_xyxy, scores_f, iou_thresh)

    # Top-K (截断到 max_det)
    if len(nms_idx) > max_det:
        nms_idx = nms_idx[:max_det]

    boxes_nms = boxes_xyxy[nms_idx]
    scores_nms = scores_f[nms_idx]
    labels_nms = labels_f[nms_idx]
    coeffs_nms = coeffs_f[nms_idx]
    num_valid = len(nms_idx)

    # 掩码解码（仅对NMS后的有效框做，节省计算）
    c, h, w = protos.shape
    protos_flat = protos.view(c, -1)                          # [32, 25600]
    masks = torch.matmul(coeffs_nms, protos_flat).view(num_valid, h, w)

    # 上采样到 img_size
    masks = masks.unsqueeze(1)                                # [M, 1, 160, 160]
    masks = F.interpolate(masks, size=(img_size, img_size),
                          mode='bilinear', align_corners=False) > 0.0
    masks = masks.squeeze(1)                                   # [M, 640, 640]

    # Pad 到固定 max_det 维
    final_masks = torch.zeros((max_det, img_size, img_size), dtype=torch.bool, device=device)
    final_boxes = torch.zeros((max_det, 4), dtype=torch.float32, device=device)
    final_scores = torch.zeros(max_det, dtype=torch.float32, device=device)
    final_labels = torch.zeros(max_det, dtype=torch.int64, device=device)

    final_masks[:num_valid] = masks
    final_boxes[:num_valid] = boxes_nms
    final_scores[:num_valid] = scores_nms
    final_labels[:num_valid] = labels_nms

    return (final_masks.unsqueeze(0),     # [1, 300, 640, 640]
            final_boxes.unsqueeze(0),     # [1, 300, 4]
            final_scores.unsqueeze(0),    # [1, 300]
            final_labels.unsqueeze(0))    # [1, 300]

postprocess_map = {
    'yolo26n-seg': yolo_seg_e2e_postprocess,
    'vajra-v1-nano-seg': yolo_seg_e2e_postprocess,
    'yolo11n-seg': yolo_seg_nms_postprocess,
    'yolov12n-seg': yolo_seg_nms_postprocess,
}

class TRTInference(object):
    def __init__(self, engine_path, device='cuda', backend='torch', max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()

        if self.backend == 'cuda':
            self.stream = cuda.Stream()
        self.time_profile = TimeProfiler()
        self.time_profile_dataset = TimeProfiler()

        name = os.path.splitext(os.path.basename(engine_path))[0]
        self.postprocess = postprocess_map.get(name, full_e2e)

    def init(self):
        self.dynamic = False

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()
        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                dynamic = True
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            if self.backend == 'cuda':
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    data = np.random.randn(*shape).astype(dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr)
                else:
                    data = cuda.pagelocked_empty(trt.volume(shape), dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr)
            else:
                data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())
        return bindings

    def run_torch(self, blob):
        for n in self.input_names:
            # print(self.bindings[n].shape)
            # print(blob[n].shape)
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape)
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}
        return outputs

    def async_run_cuda(self, blob):
        for n in self.input_names:
            cuda.memcpy_htod_async(self.bindings_addr[n], blob[n], self.stream)

        bindings_addr = [int(v) for _, v in self.bindings_addr.items()]
        self.context.execute_async_v2(bindings=bindings_addr, stream_handle=self.stream.handle)

        outputs = {}
        for n in self.output_names:
            cuda.memcpy_dtoh_async(self.bindings[n].data, self.bindings[n].ptr, self.stream)
            outputs[n] = self.bindings[n].data

        self.stream.synchronize()

        return outputs

    def __call__(self, blob):
        if self.backend == 'torch':
            return self.run_torch(blob)
        elif self.backend == 'cuda':
            return self.async_run_cuda(blob)

    def synchronize(self):
        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.backend == 'cuda':
            self.stream.synchronize()

    def warmup(self, blob, n):
        for _ in range(n):
            _ = self(blob)

    def speed(self, blob, n, nonempty_process=False):
        times = []
        self.time_profile_dataset.reset()
        for i in tqdm(range(n), desc="Running Inference", unit="iteration"):
            self.time_profile.reset()
            with self.time_profile_dataset:
                img = blob[i]
                # img['orig_target_sizes'] = img['orig_target_sizes'].unsqueeze(0)
                img['orig_target_sizes'] = torch.tensor([[640, 640]], device=blob.device)
                if img['images'] is not None:
                    img['images'] = img['image'] = img['input'] = img['images'].unsqueeze(0)
                else:
                    img['image'] = img['images'] = img['input'] = img['image'].unsqueeze(0)

            with self.time_profile:
                outputs = self(img)
                _ = self.postprocess(outputs)

            times.append(self.time_profile.total)

        # end-to-end model only
        times = sorted(times)
        if len(times) > 100 and nonempty_process:
            times = times[:100]

        avg_time = sum(times) / len(times)  # Calculate the average of the remaining times
        return avg_time

def main():
    FLAGS = parse_args()
    dataset = Dataset(FLAGS.infer_dir)
    im = torch.ones(1, 3, 640, 640).cuda()
    blob = {
            'image': im,
            'images': im,
            'input': im,
            'im_shape': torch.tensor([640, 640]).to(im.device),
            'scale_factor': torch.tensor([1, 1]).to(im.device),
            'orig_target_sizes': torch.tensor([640, 640]).to(im.device),
        }
    blob['orig_target_sizes'] = blob['orig_target_sizes'].unsqueeze(0)

    engine_files = glob.glob(os.path.join(FLAGS.models_dir, "*.engine"))
    results = []

    for engine_file in engine_files:
        print(f"Testing engine: {engine_file}")
        model = TRTInference(engine_file, max_batch_size=1, verbose=False)
        model.init()
        model.warmup(blob, 1000)
        t = []
        for _ in range(3):
            t.append(model.speed(dataset, 1000, FLAGS.busy))
        avg_latency = 1000 * torch.tensor(t).median()
        results.append((engine_file, avg_latency))
        print(f"Engine: {engine_file}, Latency: {avg_latency:.2f} ms")

        del model
        torch.cuda.empty_cache()
        time.sleep(1)

    sorted_results = sorted(results, key=lambda x: x[1])
    for engine_file, latency in sorted_results:
        print(f"Engine: {engine_file}, Latency: {latency:.2f} ms")

if __name__ == '__main__':
    main()
