
import torch
import torch.nn as nn
import torchvision
import numpy as np
import sklearn
# !pip install sklearn
import sklearn.metrics
import matcher
from tqdm import tqdm
import torchvision
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader





from typing import Dict


class CNNbackbone(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.resnet = list(torchvision.models.resnet34().children())[:-2]
        self.cuttedResnet34 = nn.Sequential(*self.resnet)
        self.cnn1 = nn.Conv2d(512, 2048, 3, 1, 1)
        self.descaler = nn.Conv2d(2048, d, 1)
    def forward(self, input: torch.Tensor):
        # image input z,c,800,600
        y = self.cuttedResnet34(input)
        y = self.cnn1(y)
        y = self.descaler(y)
        return y
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout_rate):
        super().__init__()
        # 4x as output of l1 and in l2
        # https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5208s
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer
            nn.Dropout(dropout_rate),
        )
    def forward(self, x):
        return self.net(x)
class DETR(nn.Module):
    def __init__(self, descaleDims: int, num_heads: int, num_classes: int, num_queries: int):
        super().__init__()
        self.backbone = CNNbackbone(descaleDims)
        # if (height / 32) - (height // 32) > 1e-4:
        #     hfix = (height // 32 ) + 1
        # else:
        #     hfix = height // 32
        # if (width / 32) - (width // 32) > 1e-4:
        #     wfix = (width // 32) + 1
        # else:
        #     wfix = width // 32
        # print(f"{wfix}, {hfix}")


        #self.emb_pos = nn.Embedding(descaleDims, (wfix)*(hfix))


        self.descale = descaleDims

        self.object_queries = nn.Parameter(torch.rand(num_queries, self.descale))
        self.col_embed = nn.Parameter(torch.rand(100, self.descale // 2))
        self.row_embed = nn.Parameter(torch.rand(100, self.descale // 2))

        # self.d_enc = (wfix)*(hfix)

        self.num_heads = num_heads
        # self.width = width
        # self.height = height
        self.num_classes = num_classes


        self.encoderL = nn.TransformerEncoderLayer(self.descale, self.num_heads) # cos co dzieli d równo.
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoderL, num_layers=4)

        self.decoderL = nn.TransformerDecoderLayer(self.descale, self.num_heads)
        self.decoder = nn.TransformerDecoder(self.decoderL, 4)

        self.ffn1 = FeedForward(self.descale, 0.1)
        self.ffn2 = FeedForward(self.descale, 0.1)
        self.ffn3 = FeedForward(self.descale, 0.1)
        
        self.single_box_out = nn.Linear(self.descale, 4)# X, Y _n | W, H _n
        self.categorical = nn.Linear(self.descale, num_classes + 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        pass
    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:

        W_ORG, H_ORG = input.shape[-2:]

        y = self.backbone(input).squeeze(0)
        batch_size, d, H, W = (None, None, None, None)
        if(len(y.size()) == 3):
            y = y.unsqueeze(0)
        if (len(y.size()) == 4):
            batch_size, d, H, W = y.size()
        # print(y.size())
        # y = y.reshape(d, H*W)
        # print(self.col_embed[:W])
        emb_pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        # add posistional embedding
        # print(f"{y.shape}, {emb_pos.shape}, {H}, {W}")
        # print(f"{y.flatten(2).permute(2,0,1).shape}, {emb_pos.shape}")
        h = y.flatten(2).permute(2, 0, 1)
        h = h + emb_pos
        # y = y.unsqueeze(1).flatten(2).permute(2,0,1) + emb_pos
        # print(f"{h.shape}, {emb_pos.shape}")
        y = self.encoder.forward(h)
        
        tgt = self.object_queries.unsqueeze(1).repeat(1, batch_size, 1)
        # print(f"{tgt.shape} {y.shape}")
        y = self.decoder.forward(tgt, y)
        # torch.Size([1024, 475]) d, H*W
        # dla kazdego d
        y = self.ffn3(self.ffn2(self.ffn1(y)))
        
        boxes = self.single_box_out(y)
        classes = self.categorical(y)

        return {"pred_boxes": self.prediction_normalized_xywh_to_x1y1x2y2(self.sigmoid(boxes), W_ORG, H_ORG),
                "pred_logits": classes}
    def prediction_normalized_xywh_to_x1y1x2y2(self, box: torch.Tensor, imgWidth: int, imgHeight: int):
        x_n = box[..., 0]
        y_n = box[..., 1]
        w_n = box[..., 2]
        h_n = box[..., 3]

        x1 = x_n * imgWidth
        y1 = y_n * imgHeight
        x2 = x1 + w_n * imgWidth
        y2 = y1 + h_n * imgHeight

        return torch.stack([x1, y1, x2, y2], dim=-1)





IMAGE_FIX_WIDTH = 800
IMAGE_FIX_HEIGHT = 800
# Define a simple transform (resize, to tensor, etc.)
transform = transforms.Compose([
    transforms.Resize((IMAGE_FIX_WIDTH, IMAGE_FIX_HEIGHT)),
    transforms.ToTensor()
])

# Load the training dataset
coco_train = CocoDetection(
    root='coco/train2017',
    annFile='coco/annotations/instances_train2017.json',
    transform=transform
)
def collate_fn(batch):
    """
    batch: list of tuples (image, anns)
      - image: Tensor[C,H,W]
      - anns: list of dicts, each with keys 'bbox' and 'category_id' (and others)
    Returns:
      images: Tensor[B, C, H, W]
      targets: list of B dicts, each with:
        - 'boxes': Tensor[N_i, 4] in [x1,y1,x2,y2] format
        - 'labels': Tensor[N_i]
    """
    images, all_anns = zip(*batch)
    # Stack images into [B, C, H, W]
    images = torch.stack(images, dim=0)

    targets = []
    for anns in all_anns:
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'] - 1)  # zero‑base labels

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        targets.append({
            'boxes': boxes,
            'labels': labels
        })

    return images, targets

from util.box_ops import generalized_box_iou
NUM_CLASSES = 90

def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculates IoU for corresponding pairs of bounding boxes.

    Args:
        boxes1 (torch.Tensor): Tensor of shape (N, 4), each row is (x1, y1, x2, y2).
        boxes2 (torch.Tensor): Tensor of shape (N, 4), each row is (x1, y1, x2, y2).

    Returns:
        torch.Tensor: IoU for each pair, shape (N,).
    """
    if boxes1.shape != boxes2.shape or boxes1.shape[-1] != 4:
        raise ValueError(f"Expected both inputs to have shape (N,4), got {boxes1.shape} and {boxes2.shape}")

    # Intersection coordinates
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    # Intersection area
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Areas of input boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    # Union area
    union_area = area1 + area2 - inter_area
    # Avoid division by zero
    iou = inter_area / union_area.clamp(min=1e-6)
    return iou



def train_fn(model: nn.Module, dataloader: torch.utils.data.DataLoader, epochs: int = 30):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(DEVICE)
    model.train()
    loss_clasfier = nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters())
    loss_box_matcher = matcher.HungarianMatcher(device=DEVICE).to(DEVICE)

    # every accum_steps to optim step (weight update)
    accumulation_steps = 16

    for epoch in tqdm(range(epochs), desc="Training", total=epochs):
        batch_loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (images, targets) in enumerate(batch_loop):
            images = images.to(DEVICE)
            optimizer.zero_grad()
            prediction = model.forward(images)
            # print(prediction["pred_logits"].shape, prediction["pred_boxes"].shape)


            # [(tensor_indicies_boxes, tensor_indicies_clasf)]
            # print(prediction['pred_boxes'].shape)
            prediction['pred_boxes'] = prediction['pred_boxes'].permute(1, 0, 2)
            prediction['pred_logits'] = prediction['pred_logits'].permute(1, 0, 2)
            recon = loss_box_matcher.forward(prediction, targets)
            

            pred_boxes = [prediction['pred_boxes'][k, recon[k][0], :] for k in range(prediction['pred_boxes'].shape[0])]
            tg_boxes = [targets[k]['boxes'].to(DEVICE) for k in range(len(targets))]
            
            pred_clasf = [prediction['pred_logits'][k, recon[k][1], :] for k in range(prediction['pred_logits'].shape[0])]
            tg_clasf = [targets[k]['labels'].to(DEVICE) for k in range(len(targets))]

            # print(f"iou mean: {torch.mean(generalized_box_iou(torch.cat([pred_image_boxes for pred_image_boxes in pred_boxes]), torch.cat([true_image_boxes for true_image_boxes in tg_boxes])))}")

            iou_loss = 1 - torch.mean(generalized_box_iou(torch.cat([pred_image_boxes for pred_image_boxes in pred_boxes]),\
                                                           torch.cat([true_image_boxes for true_image_boxes in tg_boxes])))
            iou_loss = iou_loss * 5
            
            one_hot_target_class = torch.nn.functional.one_hot(torch.cat([tg for tg in tg_clasf]), num_classes=NUM_CLASSES+1)
            loss_clasf = loss_clasfier.forward(torch.cat([pc for pc in pred_clasf]), one_hot_target_class.float())
            total_loss = 0.0

            total_loss += (iou_loss + loss_clasf) / accumulation_steps
            batch_loop.set_postfix({
                'Loss': f'{total_loss:.4f}',
                'GIoU': f'{iou_loss.item():.4f}',
                'Cls': f'{loss_clasf.item():.4f}'
            })
            # for i, targetValue in enumerate(targets):
            #     prediction_boxes = prediction['pred_boxes'][recon[i][0]]
            #     prediction_clasf = prediction['pred_logits'][recon[i][1]]
            #     print(f"{prediction['pred_boxes'].shape} {prediction_boxes.shape} , {targets[i]['boxes'].shape}")
                
                

            # print(f"epoch: {epoch+1}, loss: {total_loss:.6f} loss/batch_size: {(total_loss / len(targets)):.6f}")
            total_loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    model.eval()


detr = None
if detr:
    del detr
detr = DETR(1024, 8, NUM_CLASSES, 128).to('cuda')


# Load into DataLoader
train_loader = DataLoader(coco_train, batch_size=8, shuffle=True, num_workers=8, collate_fn=collate_fn)
train_fn(model=detr, dataloader=train_loader)