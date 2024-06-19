import os
import time
import math
import torch
import logging
from tqdm import tqdm
from accelerate import Accelerator
from transformers import get_scheduler
from torch.utils.tensorboard import SummaryWriter

from svd_ood.loss import LoCoOpLoss
from svd_ood.model_hub import ModelHub
from svd_ood.dataloaders import get_id_loader
from svd_ood.utils.utils import setup_seed
from svd_ood.utils.logger import setup_logger
from svd_ood.utils.data_utils import get_id_labels
from svd_ood.utils.argparser import parse_args, print_args, save_args
from peft import PeftModel, ModifiedLoraConfig
from test_ood import run_test_ood
from test_classify import run_test_classify


def run_finetune(args, verbose=True):
    assert args.split != "test", "Should not run finetune on test split"

    start_time = time.time()
    setup_seed(args.seed)
    accelerator = Accelerator()
    model_hub = ModelHub(args.model_type)
    tensorboard = SummaryWriter(log_dir=os.path.join(args.log_directory, "tensorboard"))

    ### Initialize model, preprocess, tokenizer ###
    # 1. Load the pretrained model
    if verbose:
        logging.info(f"Loading {args.model_type} model: {args.model_name}...")
    model_args = {"model_name": args.model_name, "device": args.device}
    if args.model_type == "LoCoOp":
        model_args.update({"n_ctx": args.n_ctx, "locoop_ckpt": args.locoop_ckpt, "verbose": verbose})
    model, train_preprocess, val_preprocess, tokenizer = model_hub.load(**model_args)

    # 2. Apply SVDLoRA / LoRA to the model
    if args.lora_settings:
        # Convert lora_settings to target_modules
        target_modules = model_hub.to_target_modules(model, args.lora_settings, verbose)
    elif args.target_modules:
        target_modules = args.target_modules
    else:
        raise ValueError("Either lora_settings or target_modules should be provided.")
    # 2.1 Create a LoraModel with the specified rank config
    lora_config = ModifiedLoraConfig(r=args.lora_r, target_modules=target_modules)
    model = PeftModel(model, lora_config, verbose=verbose)
    if args.lora_svd_init:
        # 2.2 Split weights with SVD to base model and lora adapter
        model.svd_init(lora_weights=args.lora_svd_init_type)  # "small", "large", "random"
        # 2.3 Disable lora scaling to maintain the original model output
        model.disable_lora_scaling()

    if args.clip_ckpt:
        logging.warning(f"clip_ckpt ({args.clip_ckpt}) is not used in finetuning, skipping...")

    # 3. Ready for finetuning
    # Move the model to the device
    model = model.to(args.device)
    # Enable dropout layers
    model.train()
    model.print_trainable_parameters()
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, list(p.size()))

    ### Prepare dataloader ###
    id_labels = get_id_labels(args.id_dataset)
    id_loader = get_id_loader(args.data_root, args.batch_size, args.id_dataset,
                              args.split, train_preprocess, shuffle=True)

    ### Prepare loss, optimizer and scheduler ###
    # 1. Loss function
    loss_fn = LoCoOpLoss(args.locoop_lambda, args.locoop_top_k)
    # 2. Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.SGD(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    model, optimizer, id_loader = accelerator.prepare(model, optimizer, id_loader)

    # 3. Scheduler and math around the number of training steps
    num_update_steps_per_epoch = math.ceil(len(id_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    ### Training ###
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if verbose:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(id_loader.dataset)}")
        logging.info(f"  Num Epochs = {args.num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    if args.model_type != "SwinTransformerV2":
        texts = tokenizer([f"a photo of a {c}" for c in id_labels],
            return_tensors="pt", padding=True, truncation=True).to(model.device)
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, (images, labels) in enumerate(id_loader):
            # forward
            images = images.to(model.device)
            labels = labels.to(model.device)
            if args.model_type == "SwinTransformerV2":
                outputs = model(images)
                logits_global = outputs.logits
                logits_local = outputs.logits_local
            else:
                outputs = model(pixel_values=images,
                                input_ids=texts.input_ids,
                                attention_mask=texts.attention_mask)
                logits_global = outputs.logits_per_image      # with logit_scale.exp()
                logits_local = outputs.logits_per_image_local # with logit_scale.exp()
            bs, weight, height, cls = logits_local.shape
            logits_local = logits_local.view(bs, weight*height, cls)
            loss_dict = loss_fn.cal_loss(logits_global, logits_local, labels)
            # log to tensorboard
            tensorboard.add_scalar("train/locoop_loss", loss_dict["locoop_loss"], completed_steps)
            tensorboard.add_scalar("train/loss_id", loss_dict["loss_id"], completed_steps)
            tensorboard.add_scalar("train/loss_ood", loss_dict["loss_ood"], completed_steps)
            tensorboard.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], completed_steps)
            if verbose and step % args.logging_steps == 0:
                try:
                    logging.info(f"Epoch {epoch} | Step {step} | Loss: {loss_dict['locoop_loss']:.4f} | "
                             f"Loss ID: {loss_dict['loss_id']:.4f} | Loss OOD: {loss_dict['loss_ood']:.4f}")
                except:
                    logging.info(f"Epoch {epoch} | Step {step} | Loss: {loss_dict['locoop_loss'].item():.4f} | "
                             f"Loss ID: {loss_dict['loss_id'].item():.4f} | Loss OOD: {loss_dict['loss_ood'].item():.4f}")
            # backward
            loss = loss_dict["locoop_loss"]
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(id_loader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

    ### Merge and save the model ###
    merged_model = model.merge_and_unload()
    state_dict = merged_model.state_dict()
    torch.save(state_dict, os.path.join(args.log_directory, "finetuned_model.pth"))
    if verbose:
        logging.info(f"Model saved to {args.log_directory}/finetuned_model.pth")

    end_time = time.time()
    t = end_time - start_time
    if verbose:
        logging.info(f"############ Done! Finetune time: {t//60:.0f}m {t%60:.0f}s ############")

if __name__ == "__main__":
    args = parse_args(finetune=True)
    setup_logger(log_dir=args.log_directory, log_file="finetune.log")
    save_args(args, args.log_directory, "config_finetune.json")
    print_args(args)
    run_finetune(args)

    ### run test on the finetuned model ###
    # set args for test
    args.split = "test"
    args.config_file = None
    args.exp_name = "Run test on the finetuned model"
    args.lora_svd_init = False  # we have already merged the model
    args.clip_ckpt = os.path.join(args.log_directory, "finetuned_model.pth")
    if "loss" in args.scorers:
        args.scorers.remove("loss")
    # delete finetune args
    del args.num_train_epochs
    del args.max_train_steps
    del args.gradient_accumulation_steps
    del args.learning_rate
    del args.weight_decay
    del args.momentum
    del args.lr_scheduler_type
    del args.num_warmup_steps
    del args.locoop_lambda
    del args.locoop_top_k
    del args.logging_steps
    del args.lora_svd_init_type
    del args.lora_settings
    setup_logger(log_dir=args.log_directory, log_file="test.log")
    save_args(args, args.log_directory, "config_test.json")
    run_test_ood(args)
    run_test_classify(args)