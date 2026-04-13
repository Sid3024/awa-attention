import torch
import torch.nn as nn
import torch.nn.functional as F


from src.model.v1.model import SwinTransformer
from src.log.setup import setup_logger
from src.data_io.data_loader import build_loader
from src.config.config import model_config, hyper_param_config, data_config, run_config, my_config
from src.utils.train_helpers import get_lr


LOG_FILE_PATH = run_config.LOG_FILE_PATH
MODEL_FILE_PATH = run_config.MODEL_FILE_PATH

logger = setup_logger(LOG_FILE_PATH)

logger.info("start run")
logger.info(f"{data_config=}")
logger.info(f"{run_config=}")
logger.info(f"{hyper_param_config=}")
logger.info(f"{model_config=}")

torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"
autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float16
model = SwinTransformer(num_classes=run_config.num_classes)
model.to(device)
logger.info(f"{device=}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"{total_params=}")

train_dataset, val_dataset, train_loader, val_loader, num_classes = build_loader(data_config)
logger.info(f"Num classes: {num_classes}")
logger.info(f"Train size: {len(train_dataset)}")
logger.info(f"Val size: {len(val_dataset)}")

assert hyper_param_config.total_batch_size % data_config.gpu_batch_size == 0
total_batch_size = hyper_param_config.total_batch_size # used for alphazero
batch_size = data_config.gpu_batch_size
grad_accum_steps = total_batch_size // batch_size


adam_lr = hyper_param_config.constant_lr if hyper_param_config.constant_lr is not None else 0.0
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=adam_lr,
    weight_decay=hyper_param_config.adamw_weight_decay,
)

def train(model, train_loader, val_loader, optimizer):
    model.train()
    train_iter = iter(train_loader)
    logger.info("Starting training")
    for step in range(hyper_param_config.total_train_steps):
        optimizer.zero_grad(set_to_none=True)
        if hyper_param_config.constant_lr is None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = get_lr(step)
        step_loss = 0 # for logging only
        for micro_step in range(grad_accum_steps):
            try:
                image_tensor, label_tensor = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                image_tensor, label_tensor = next(train_iter)
            image_tensor, label_tensor = image_tensor.to(device), label_tensor.to(device)
            
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                pred_tensor = model(image_tensor)
                raw_loss = F.cross_entropy(input=pred_tensor, target=label_tensor)
            
            loss = raw_loss / grad_accum_steps
            step_loss += raw_loss.item() 
            loss.backward()
        optimizer.step()

        current_lr = hyper_param_config.constant_lr if hyper_param_config.constant_lr is not None else get_lr(step)

        logger.info(f"step: {step} | loss: {step_loss / grad_accum_steps} | lr: {current_lr}")

        if step % run_config.model_save_interval == 0 or step == hyper_param_config.total_train_steps - 1:
            if MODEL_FILE_PATH is not None:
                torch.save(model.state_dict(), MODEL_FILE_PATH)
                logger.info(f"Model parameters saved to {MODEL_FILE_PATH} at step {step}")
        
        if (my_config.run_validation and step % run_config.val_interval == 0 and step != 0) or step == hyper_param_config.total_train_steps - 1:
            val(model, val_loader, step)

def val(model, val_loader, train_step):
    model.eval()
    val_iter = iter(val_loader)
    logger.info("Starting validation")
    gpu_step = 0
    loss_list = []
    with torch.no_grad():
        while True:
            try:
                image_tensor, label_tensor = next(val_iter)
            except StopIteration:
                break
            image_tensor, label_tensor = image_tensor.to(device), label_tensor.to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                pred_tensor = model(image_tensor)
                loss = F.cross_entropy(input=pred_tensor, target=label_tensor)
            logger.info(f"Validation step: {gpu_step} | loss: {loss.item()}")
            loss_list.append(loss.item())
            gpu_step += 1
        val_loss = sum(loss_list)/len(loss_list)
        logger.info(f"Validation Completed. {val_loss=} at {train_step=}")
        
    model.train()
        

if __name__ == "__main__":
    if my_config.run_training:
        train(model, train_loader, val_loader, optimizer)

