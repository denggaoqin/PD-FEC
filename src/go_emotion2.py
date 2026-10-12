import os
import math
import random
import numpy as np
import torch
import pandas as pd

from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# --------------------------
# Environment and AMP
# --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# --------------------------
# Configuration
# --------------------------
MODEL = "./data"
train_path = "./data"
valid_path = "./data"
test_path  = "./data"

seed = 42


binary_epochs = 1
lr_binary = 2e-5 


multi_epochs = 7 
lr_multi = 1.3e-5                
weight_decay = 0.005             
warmup_ratio = 0.051             
max_len = 96                   
batch_size = 16                 
grad_accum_steps = 1            


freeze_top_k = 4
freeze_epochs = 4              


use_focal = True
focal_gamma = 2.5           
focal_weight = 0.21             



use_peak_module = True
peak_loss_weight = 0.026 
peak_warmup_epochs = 2
peak_start_epoch = 2
peak_temperature = 0.7
peak_sharpness = 2.0
peak_contrast_margin = 0.05
peak_entropy_weight = 0.01


use_confusion_graph_reg = True
conf_graph_weight = 0.010 
conf_graph_momentum = 0.9
conf_graph_topk = 6
conf_graph_warmup_epochs = 2
conf_graph_start_epoch = 2
conf_graph_conf_threshold = 0.35
conf_graph_hard_threshold = 0.65
conf_graph_loss_cap = 0.35

# CosFace 
use_cosface = True
cosface_s = 20.0              
cosface_m = 0.20               
cosface_dropout = 0.1            

# Early stopping
use_early_stopping = True
patience = 3                   
save_best_path = "best_multiclass.pt"

# --------------------------
# Set random seed
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(seed)

# --------------------------
# Load dataset
# --------------------------
df_train = pd.read_csv(train_path)
df_valid = pd.read_csv(valid_path)
df_test  = pd.read_csv(test_path)

# --------------------------
# GoEmotion positive labels
# --------------------------
positive_labels = {0, 1, 4, 5, 7, 8, 13, 15, 17, 18, 20, 21, 22, 23, 26}

def get_binary_label(x):
    return 1 if x in positive_labels else 0

df_train["binary"] = df_train["label"].apply(get_binary_label)
df_valid["binary"] = df_valid["label"].apply(get_binary_label)
df_test["binary"]  = df_test["label"].apply(get_binary_label)

# --------------------------
# Dataset
# --------------------------
class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=96):
        self.texts = df["text"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.binary_labels = df["binary"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "binary": torch.tensor(self.binary_labels[idx], dtype=torch.long),
        }

# --------------------------
# CosFace Head（Optional）
# --------------------------
class CosFaceHead(nn.Module):
    def __init__(self, hidden_size, num_classes, s=20.0, m=0.20, dropout=0.1):
        super().__init__()
        self.s = s
        self.m = m
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(num_classes, hidden_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        features = self.dropout(features)
        cosine = nn.functional.linear(
            nn.functional.normalize(features, dim=1),
            nn.functional.normalize(self.weight, dim=1)
        )

        if labels is not None:
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            logits = cosine - one_hot * self.m
        else:
            logits = cosine

        return logits * self.s


class TokenPeakModule(nn.Module):
    def __init__(
        self,
        hidden_size,
        temperature=0.7,
        sharpness=2.0,
        contrast_margin=0.05,
        entropy_weight=0.01,
        dropout=0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.sharpness = sharpness
        self.contrast_margin = contrast_margin
        self.entropy_weight = entropy_weight
        mid_size = max(64, hidden_size // 4)

        self.score_net = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(mid_size, 1),
        )
        self.fusion_gate = nn.Linear(hidden_size * 3, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask):
        token_mask = attention_mask.float()
        content_mask = token_mask.clone()
        content_mask[:, 0] = 0.0

        has_content = content_mask.sum(dim=-1, keepdim=True) > 0
        candidate_mask = torch.where(has_content, content_mask, token_mask)

        left_mask = torch.zeros_like(candidate_mask)
        right_mask = torch.zeros_like(candidate_mask)
        left_mask[:, 1:] = token_mask[:, :-1]
        right_mask[:, :-1] = token_mask[:, 1:]
        local_mask = candidate_mask * left_mask * right_mask

        raw_scores = self.score_net(hidden_states).squeeze(-1)
        scores = torch.tanh(raw_scores)

        left_scores = torch.zeros_like(scores)
        right_scores = torch.zeros_like(scores)
        left_scores[:, 1:] = scores[:, :-1]
        right_scores[:, :-1] = scores[:, 1:]
        neighbor_mean = 0.5 * (left_scores + right_scores)

        local_contrast = scores - neighbor_mean
        peak_strength = nn.functional.softplus(local_contrast / max(self.temperature, 1e-6))

        has_local_peak = local_mask.sum(dim=-1, keepdim=True) > 0
        peak_logits = scores + self.sharpness * peak_strength
        peak_logits = peak_logits.masked_fill(local_mask < 0.5, -1e4)
        fallback_logits = scores.masked_fill(candidate_mask < 0.5, -1e4)
        peak_logits = torch.where(has_local_peak, peak_logits, fallback_logits)

        peak_attn = torch.softmax(peak_logits, dim=-1)
        peak_context = torch.bmm(peak_attn.unsqueeze(1), hidden_states).squeeze(1)

        cls = hidden_states[:, 0, :]
        gate_input = torch.cat([cls, peak_context, torch.abs(cls - peak_context)], dim=-1)
        fusion_gate = torch.sigmoid(self.fusion_gate(gate_input))
        fused_cls = self.norm(cls + fusion_gate * self.dropout(peak_context))

        active_peak_attn = peak_attn * local_mask
        active_denom = active_peak_attn.sum(dim=-1).clamp_min(1e-6)
        weighted_contrast = (active_peak_attn * local_contrast).sum(dim=-1) / active_denom
        contrast_loss = nn.functional.softplus(self.contrast_margin - weighted_contrast).mean()

        valid_count = candidate_mask.sum(dim=-1).clamp_min(2.0)
        entropy = -(peak_attn * torch.log(peak_attn.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / torch.log(valid_count)
        peak_loss = contrast_loss + self.entropy_weight * entropy.mean()

        return {
            "fused_cls": fused_cls,
            "peak_loss": peak_loss,
            "peak_attn": peak_attn,
            "peak_strength": peak_strength,
        }

# --------------------------
# Two-stage model
# --------------------------
class TwoStageEmotionModel(nn.Module):
    def __init__(
        self,
        model_path,
        num_labels=27,
        use_peak_module=False,
        use_cosface=False,
        cosface_s=20.0,
        cosface_m=0.20,
        cosface_dropout=0.1,
        peak_temperature=0.7,
        peak_sharpness=2.0,
        peak_contrast_margin=0.05,
        peak_entropy_weight=0.01,
    ):
        super().__init__()
        self.use_peak_module = use_peak_module
        self.use_cosface = use_cosface
        self.binary_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.multi_model  = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

        hidden_size = self.multi_model.config.hidden_size

        if self.use_peak_module:
            self.peak_module = TokenPeakModule(
                hidden_size=hidden_size,
                temperature=peak_temperature,
                sharpness=peak_sharpness,
                contrast_margin=peak_contrast_margin,
                entropy_weight=peak_entropy_weight,
                dropout=cosface_dropout,
            )
        else:
            self.peak_module = None

        if self.use_cosface:
            self.cosface_head = CosFaceHead(
                hidden_size=hidden_size,
                num_classes=num_labels,
                s=cosface_s,
                m=cosface_m,
                dropout=cosface_dropout
            )
        else:
            self.cosface_head = None

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        stage="binary",
        return_peak=False,
        use_cosface=False
    ):
        need_peak = self.use_peak_module and self.peak_module is not None
        need_cosface = (stage == "multi" and self.use_cosface and use_cosface)

        if stage == "binary":
            return self.binary_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        
        if need_peak or need_cosface:
            out = self.multi_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                output_hidden_states=True,
                return_dict=True
            )
            sequence_output = out.hidden_states[-1]
            cls = sequence_output[:, 0, :]

            result = {}
            if need_peak:
                peak_out = self.peak_module(sequence_output, attention_mask)
                cls = peak_out["fused_cls"]
                if return_peak:
                    result["peak_loss"] = peak_out["peak_loss"]
                    result["peak_attn"] = peak_out["peak_attn"]
                    result["peak_strength"] = peak_out["peak_strength"]

            if need_cosface:
                logits = self.cosface_head(cls, labels=labels)
            elif need_peak and hasattr(self.multi_model, "classifier"):
                fused_sequence = sequence_output.clone()
                fused_sequence[:, 0, :] = cls
                logits = self.multi_model.classifier(fused_sequence)
            else:
                logits = out.logits

            result["logits"] = logits
            return result

        return self.multi_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

# --------------------------
# Focal Loss（Optional）
# --------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce_none = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce = self.ce_none(logits, targets)
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal

class ConfusionGraphRegularizer(nn.Module):
    def __init__(self, num_classes, momentum=0.9, topk=6, eps=1e-6, conf_threshold=0.35):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.topk = topk
        self.eps = eps
        self.conf_threshold = conf_threshold
        self.register_buffer("conf_graph", torch.zeros(num_classes, num_classes))
        self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def update_graph(self, logits, targets):
        probs = torch.softmax(logits.detach(), dim=-1)
        conf = probs.max(dim=-1).values
        keep = conf >= self.conf_threshold
        if keep.sum().item() == 0:
            return

        probs = probs[keep]
        targets = targets[keep]
        conf_batch = torch.zeros(
            self.num_classes, self.num_classes,
            device=logits.device, dtype=probs.dtype
        )
        row_count = torch.zeros(self.num_classes, device=logits.device, dtype=probs.dtype)
        ones = torch.ones_like(targets, dtype=probs.dtype)
        row_count.index_add_(0, targets, ones)
        conf_batch.index_add_(0, targets, probs)
        conf_batch = conf_batch / row_count.unsqueeze(1).clamp_min(1.0)

        if bool(self.is_initialized):
            self.conf_graph.mul_(self.momentum).add_((1.0 - self.momentum) * conf_batch)
        else:
            self.conf_graph.copy_(conf_batch)
            self.is_initialized.fill_(True)

        self.conf_graph.fill_diagonal_(0.0)
        row_sum = self.conf_graph.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        self.conf_graph.div_(row_sum)

        if self.topk is not None and self.topk > 0 and self.topk < self.num_classes:
            top_vals, top_idx = torch.topk(self.conf_graph, k=self.topk, dim=-1)
            sparse = torch.zeros_like(self.conf_graph)
            sparse.scatter_(1, top_idx, top_vals)
            sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            self.conf_graph.copy_(sparse)

    def forward(self, logits, targets, update_graph=True, hard_threshold=0.65):
        if update_graph:
            self.update_graph(logits, targets)
        probs = torch.softmax(logits, dim=-1)
        true_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        hard_mask = (true_probs <= hard_threshold).float()
        conf_targets = self.conf_graph[targets]
        reg_per_sample = (probs * conf_targets).sum(dim=-1)
        denom = hard_mask.sum().clamp_min(1.0)
        return (reg_per_sample * hard_mask).sum() / denom

# --------------------------
# class weights
# --------------------------
def compute_class_weights_log(labels, num_classes, eps=1e-6):
    counts = [0] * num_classes
    for l in labels:
        counts[int(l)] += 1
    freqs = torch.tensor(counts, dtype=torch.float) + eps
    weights = 1.0 / torch.log(freqs + 1.0)
    weights = weights / weights.sum() * num_classes
    return weights


def freeze_encoder_except_topk(base_model, top_k=2, freeze=True):

    if not hasattr(base_model, "encoder") or not hasattr(base_model.encoder, "layer"):
        for p in base_model.parameters():
            p.requires_grad = not freeze
        return

    layers = list(base_model.encoder.layer)
    n = len(layers)
    for i, layer in enumerate(layers):
        if freeze:
            should_require = (i >= n - top_k)
        else:
            should_require = True
        for p in layer.parameters():
            p.requires_grad = should_require

def set_head_trainable(model, stage="multi", trainable=True):
    if stage == "multi":
        head = model.multi_model.classifier
    else:
        head = model.binary_model.classifier
    for p in head.parameters():
        p.requires_grad = trainable

# --------------------------
# transfer encoder weights
# --------------------------
def transfer_encoder_weights(model):
    try:
        src = model.binary_model.base_model
        dst = model.multi_model.base_model
        dst.load_state_dict(src.state_dict(), strict=False)
        print(">> transferred encoder weights")
    except Exception as e:
        print(f">> transfer skipped: {e}")

# --------------------------
# eval
# --------------------------
@torch.no_grad()
def evaluate_multiclass(model, dataloader, device):
    model.eval()
    all_labels, all_preds = [], []
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        out = model(input_ids=ids, attention_mask=mask, stage="multi", use_cosface=use_cosface)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        preds = logits.argmax(dim=-1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    wf1 = f1_score(all_labels, all_preds, average="weighted")
    mf1 = f1_score(all_labels, all_preds, average="macro")
    return acc, wf1, mf1

@torch.no_grad()
def evaluate_binary(model, dataloader, device):
    model.eval()
    all_labels, all_preds = [], []
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["binary"].to(device)

        out = model(input_ids=ids, attention_mask=mask, stage="binary")
        preds = out.logits.argmax(dim=-1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
    return accuracy_score(all_labels, all_preds)

# --------------------------
# Train one epoch
# --------------------------
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    stage="binary",
    grad_accum_steps=1,
    ce_loss_fn=None,
    focal_loss_fn=None,
    focal_w=0.2,
    use_focal=False,
    epoch_idx=0,
    use_peak_module=False,
    peak_aux_weight=0.0,
    conf_graph_reg=None,
    conf_graph_weight=0.0,
    peak_warmup_epochs=0,
    peak_start_epoch=1,
    conf_graph_warmup_epochs=0,
    conf_graph_start_epoch=1,
    conf_graph_hard_threshold=0.65,
    conf_graph_loss_cap=1.0,
    use_cosface=False
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_peak_loss = 0.0
    steps = 0
    peak_steps = 0
    opt_steps = 0
    peak_w_base = peak_aux_weight * min(1.0, float(epoch_idx + 1) / max(1, peak_warmup_epochs))
    peak_w = peak_w_base if (epoch_idx + 1) >= peak_start_epoch else 0.0
    conf_w_base = conf_graph_weight * min(1.0, float(epoch_idx + 1) / max(1, conf_graph_warmup_epochs))
    conf_w = conf_w_base if (epoch_idx + 1) >= conf_graph_start_epoch else 0.0

    pbar = tqdm(dataloader, desc=f"Train({stage})", ncols=120)
    for step_idx, batch in enumerate(pbar, start=1):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        labels = batch["binary" if stage == "binary" else "label"].to(device)
        need_peak = (stage == "multi" and use_peak_module and peak_aux_weight > 0)

        with autocast():
            
            out = model(
                input_ids=ids,
                attention_mask=mask,
                labels=labels if (stage == "multi" and use_cosface) else None,
                stage=stage,
                return_peak=need_peak,
                use_cosface=use_cosface
            )
            logits = out["logits"] if isinstance(out, dict) else out.logits

            if stage == "binary":
               
                loss = nn.CrossEntropyLoss()(logits, labels)
            else:
                ce = ce_loss_fn(logits, labels)
                if use_focal and focal_loss_fn is not None:
                    focal = focal_loss_fn(logits, labels)
                    loss = ce + focal_w * focal
                else:
                    loss = ce
                if need_peak and peak_w > 0 and isinstance(out, dict) and "peak_loss" in out:
                    peak_loss = out["peak_loss"]
                    loss = loss + peak_w * peak_loss
                    total_peak_loss += float(peak_loss.detach().cpu())
                    peak_steps += 1
                if conf_graph_reg is not None and conf_w > 0:
                    conf_reg = conf_graph_reg(
                        logits=logits,
                        targets=labels,
                        update_graph=True,
                        hard_threshold=conf_graph_hard_threshold
                    )
                    conf_reg = torch.clamp(conf_reg, max=conf_graph_loss_cap)
                    loss = loss + conf_w * conf_reg

            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        total_loss += float(loss.detach().cpu())
        steps += 1

        if step_idx % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            opt_steps += 1
            if scheduler is not None:
                scheduler.step()

        avg_loss = total_loss / max(1, steps)
        postfix = {"loss": avg_loss, "opt_steps": opt_steps, "peak_w": round(peak_w, 4), "conf_w": round(conf_w, 4)}
        if peak_steps > 0:
            postfix["peak_loss"] = round(total_peak_loss / peak_steps, 4)
        pbar.set_postfix(postfix)

  
    if (len(dataloader) % grad_accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

# --------------------------
# main
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

train_ds = EmotionDataset(df_train, tokenizer, max_len=max_len)
valid_ds = EmotionDataset(df_valid, tokenizer, max_len=max_len)
test_ds  = EmotionDataset(df_test,  tokenizer, max_len=max_len)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

model = TwoStageEmotionModel(
    MODEL,
    num_labels=27,
    use_peak_module=use_peak_module,
    use_cosface=use_cosface,
    cosface_s=cosface_s,
    cosface_m=cosface_m,
    cosface_dropout=cosface_dropout,
    peak_temperature=peak_temperature,
    peak_sharpness=peak_sharpness,
    peak_contrast_margin=peak_contrast_margin,
    peak_entropy_weight=peak_entropy_weight,
).to(device)

print("\n====== Optional Modules ======")
print("use_peak_module         :", use_peak_module)
print("use_confusion_graph_reg :", use_confusion_graph_reg)
print("use_cosface             :", use_cosface)
if use_peak_module:
    print("peak_loss_weight        :", peak_loss_weight)
    print("peak_temperature        :", peak_temperature)
    print("peak_sharpness          :", peak_sharpness)
    print("peak_contrast_margin    :", peak_contrast_margin)
    print("peak_entropy_weight     :", peak_entropy_weight)

# --------------------------
# Stage 1: Binary
# --------------------------
print("\n=== Stage 1: Binary pretraining ===")
bin_optimizer = torch.optim.AdamW(
    model.binary_model.parameters(),
    lr=lr_binary,
    weight_decay=weight_decay
)


bin_total_opt_steps = math.ceil(len(train_loader) / grad_accum_steps) * max(1, binary_epochs)
bin_warmup_steps = int(warmup_ratio * bin_total_opt_steps)
bin_scheduler = get_linear_schedule_with_warmup(bin_optimizer, bin_warmup_steps, bin_total_opt_steps)

for e in range(binary_epochs):
    train_one_epoch(
        model, train_loader,
        bin_optimizer, bin_scheduler,
        device,
        stage="binary",
        grad_accum_steps=grad_accum_steps
    )
    acc_bin = evaluate_binary(model, valid_loader, device)
    print(f"[Binary Epoch {e+1}] Val ACC={acc_bin:.4f}")

# --------------------------
# transfer weights
# --------------------------
transfer_encoder_weights(model)

# --------------------------
# Stage 2: Multi-class fine-tune
# --------------------------
print("\n=== Stage 2: Multi-class fine-tune ===")

num_classes = 27
class_weights = compute_class_weights_log(df_train["label"].tolist(), num_classes).to(device)
ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

focal_loss_fn = FocalLoss(gamma=focal_gamma) if use_focal else None
conf_graph_reg = (
    ConfusionGraphRegularizer(
        num_classes=num_classes,
        momentum=conf_graph_momentum,
        topk=conf_graph_topk,
        conf_threshold=conf_graph_conf_threshold
    ).to(device)
    if use_confusion_graph_reg else None
)

if use_peak_module:
    print(">> token peak module enabled: learns peak-aware context from token hidden states")


base = model.multi_model.base_model
freeze_encoder_except_topk(base, top_k=freeze_top_k, freeze=True)
set_head_trainable(model, stage="multi", trainable=True)
print(f">> Frozen encoder layers except top {freeze_top_k} (head trainable)")


multi_params = list(model.multi_model.parameters())
if use_peak_module and model.peak_module is not None:
    multi_params += list(model.peak_module.parameters())
if use_cosface and model.cosface_head is not None:
    multi_params += list(model.cosface_head.parameters())
multi_optimizer = torch.optim.AdamW(multi_params, lr=lr_multi, weight_decay=weight_decay)

multi_total_opt_steps = math.ceil(len(train_loader) / grad_accum_steps) * max(1, multi_epochs)
multi_warmup_steps = int(warmup_ratio * multi_total_opt_steps)

multi_scheduler = get_linear_schedule_with_warmup(
    multi_optimizer,
    multi_warmup_steps,
    multi_total_opt_steps
)

best_mf1 = -1.0
bad_epochs = 0

for epoch in range(multi_epochs):

    if epoch == freeze_epochs:
        freeze_encoder_except_topk(base, top_k=freeze_top_k, freeze=False)
        print(f">> Unfroze ALL encoder layers at epoch {epoch+1}")

    train_one_epoch(
        model, train_loader,
        multi_optimizer, multi_scheduler,
        device,
        stage="multi",
        grad_accum_steps=grad_accum_steps,
        ce_loss_fn=ce_loss_fn,
        focal_loss_fn=focal_loss_fn,
        focal_w=focal_weight,
        use_focal=use_focal,
        epoch_idx=epoch,
        use_peak_module=use_peak_module,
        peak_aux_weight=peak_loss_weight,
        conf_graph_reg=conf_graph_reg,
        conf_graph_weight=conf_graph_weight if use_confusion_graph_reg else 0.0,
        peak_warmup_epochs=peak_warmup_epochs,
        peak_start_epoch=peak_start_epoch,
        conf_graph_warmup_epochs=conf_graph_warmup_epochs,
        conf_graph_start_epoch=conf_graph_start_epoch,
        conf_graph_hard_threshold=conf_graph_hard_threshold,
        conf_graph_loss_cap=conf_graph_loss_cap,
        use_cosface=use_cosface
    )

    val_acc, val_wf1, val_mf1 = evaluate_multiclass(model, valid_loader, device)
    print(f"[Multi Epoch {epoch+1}] Val ACC={val_acc:.4f} WF1={val_wf1:.4f} MF1={val_mf1:.4f}")

   
    if val_mf1 > best_mf1 + 1e-6:
        best_mf1 = val_mf1
        bad_epochs = 0
        ckpt = {"multi_model": model.multi_model.state_dict()}
        if use_peak_module and model.peak_module is not None:
            ckpt["peak_module"] = model.peak_module.state_dict()
        if use_cosface and model.cosface_head is not None:
            ckpt["cosface_head"] = model.cosface_head.state_dict()
        torch.save(ckpt, save_best_path)
        print(f">> Saved best multi_model to: {save_best_path} (best MF1={best_mf1:.4f})")
    else:
        bad_epochs += 1

    if use_early_stopping and bad_epochs >= patience:
        print(f">> Early stopping triggered (no MF1 improvement for {patience} epochs).")
        break

# --------------------------
# Final test
# --------------------------
print("\n=== Load best checkpoint and evaluate on test ===")
if os.path.exists(save_best_path):
    ckpt = torch.load(save_best_path, map_location=device)
    if isinstance(ckpt, dict) and "multi_model" in ckpt:
        model.multi_model.load_state_dict(ckpt["multi_model"])
        if model.peak_module is not None and "peak_module" in ckpt:
            model.peak_module.load_state_dict(ckpt["peak_module"])
        if model.cosface_head is not None and "cosface_head" in ckpt:
            model.cosface_head.load_state_dict(ckpt["cosface_head"])
    else:
  
        model.multi_model.load_state_dict(ckpt)
    print(f">> Loaded: {save_best_path}")
else:
    print(">> No best checkpoint found, evaluating current model.")

test_acc, test_wf1, test_mf1 = evaluate_multiclass(model, test_loader, device)

print("\n====== Final Test Results ======")
print(f"Test ACC         : {test_acc:.4f}")
print(f"Test Weighted-F1 : {test_wf1:.4f}")
print(f"Test Macro-F1    : {test_mf1:.4f}")
