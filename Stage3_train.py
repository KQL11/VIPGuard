'''
@File    :   Stage3_train_en.py
@Author  :   Kaiqing.Lin
@Update  :   2025/05/01
'''
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from termcolor import cprint
from PIL import Image

# Configuration constants
DEFAULT_CONFIG = {
    'MAX_IMAGE_SIZE': 448,        # Maximum image size
    'DEFAULT_LR': 1.0,            # Default learning rate
    'WEIGHT_DECAY': 1e-3,         # Weight decay
    'MAX_NEW_TOKENS': 4096,       # Maximum newly generated tokens
    'PRETRAIN_PATH': './checkpoints/checkpoints_attr_Stage2_merge',  # Pretrained model path
}


def train_one_epoch(model, processor, train_loader, optimizer, args, epoch, scheduler, save_base_path):
    """
    Train the model for one epoch

    Args:
        model: VIP model wrapper
        processor: processor for text, images and videos
        train_loader: training data loader
        optimizer: model optimizer
        args: training arguments
        epoch: current epoch number
        scheduler: learning rate scheduler
        save_base_path: path to save model checkpoints
    """
    from qwen_vl_utils import process_vision_info

    # Initialize training variables
    total_loss = 0
    gradient_accumulation_step = 0

    # Initialize gradient scaler and dtype for mixed precision training
    if args.use_mixed_precision:
        # Determine mixed precision dtype based on user choice
        if args.mixed_precision_dtype == 'bf16':
            mixed_precision_dtype = torch.bfloat16
            print("Enabled mixed precision training (BF16)")
        else:  # fp16
            mixed_precision_dtype = torch.float16
            print("Enabled mixed precision training (FP16)")

        # Note: bf16 often does not require gradient scaling because of its wider numeric range
        if args.mixed_precision_dtype == 'bf16':
            scaler = None  # bf16 commonly doesn't need gradient scaling
        else:
            scaler = torch.cuda.amp.GradScaler()  # fp16 needs gradient scaling
    else:
        scaler = None
        mixed_precision_dtype = torch.float32
        print("Using standard precision training (FP32)")

    # Training loop
    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
        model.train()

        # Extract data from batch
        message = data['message']
        question = data['question']
        message_ori = data['message_ori']
        question_ori = data['question_ori']

        # Extract the answer from the message
        answers = message[0].split("<|im_start|>assistant\n")[1]

        # Process image paths (flatten nested lists)
        message_ori[0]['content'][0]['image'] = message_ori[0]['content'][0]['image'][0]
        question_ori[0]['content'][0]['image'] = question_ori[0]['content'][0]['image'][0]
        img_dir = message_ori[0]['content'][0]['image']

        # Process visual inputs (images and videos)
        image_inputs, video_inputs = process_vision_info(message_ori)

        # If the image is larger than the maximum size, resize it (preserve aspect ratio)
        for i in range(len(image_inputs)):
            h, w = image_inputs[i].size
            if max(h, w) > DEFAULT_CONFIG['MAX_IMAGE_SIZE']:
                scale = DEFAULT_CONFIG['MAX_IMAGE_SIZE'] / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image_inputs[i] = image_inputs[i].resize((new_w, new_h), Image.LANCZOS)

        # If inputs are lists, take the first element
        if isinstance(message, list):
            message = message[0]
        if isinstance(question, list):
            question = question[0]

        # Prepare inputs for the full message (context + question + answer)
        inputs = processor(
            text=[message],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            our_token=True,
            our_token_length=1,
            face_pad=True,
            face_length=model.vl_model.facechecker.face_checker.vip_prompt.data.shape[0]
        )
        msg_len = inputs["input_ids"].shape[1]

        # Prepare inputs containing only the question (for generation)
        inputs_question = processor(
            text=[question],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            our_token=True,
            our_token_length=1,
            face_pad=True,
            face_length=model.vl_model.facechecker.face_checker.vip_prompt.data.shape[0]
        )

        # Process answer tokens to compute labels
        answers = processor(
            text=[answers],
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True
        )
        ans_len = answers["input_ids"].shape[1]

        # Prepare labels for loss computation
        # Compute loss only on answer tokens, mask question tokens
        labels = inputs["input_ids"].clone()
        labels[:, :msg_len - ans_len] = -100  # mask question tokens

        # Move inputs to GPU
        inputs = inputs.to("cuda")
        inputs_question = inputs_question.to("cuda")

        # Forward pass (support mixed precision)
        if args.use_mixed_precision:
            with torch.cuda.amp.autocast(dtype=mixed_precision_dtype):
                outs = model(labels=labels, **inputs)
                loss_vqa = outs.loss
                loss = loss_vqa
        else:
            outs = model(labels=labels, **inputs)
            loss_vqa = outs.loss
            loss = loss_vqa

        # Log training progress
        print(f"ID {args.name}, Loss VQA: {loss_vqa:.4f}, Img Dir: {img_dir}")

        # Backward pass (support mixed precision)
        if args.use_mixed_precision and scaler is not None:
            # FP16: use gradient scaling for backward
            scaler.scale(loss).backward()
        else:
            # BF16 or standard precision: direct backward
            loss.backward()

        # Update parameters based on gradient accumulation
        if gradient_accumulation_step >= args.gradient_accumulation_step:
            # Monitor changes in the VIP prompt
            ori_prompt = model.vl_model.facechecker.face_checker.vip_prompt.data.clone()

            # Parameter update (support mixed precision)
            if args.use_mixed_precision and scaler is not None:
                # FP16: update using the scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # BF16 or standard precision: direct optimizer step
                optimizer.step()

            new_prompt = model.vl_model.facechecker.face_checker.vip_prompt.data.clone()

            # Compute and log prompt norm changes (Just for debugging)
            ori_norm = torch.norm(ori_prompt, p=2)
            new_norm = torch.norm(new_prompt, p=2)
            diff_norm = torch.norm(new_prompt - ori_prompt, p=2)

            print(f"Original prompt norm: {ori_norm.item():.6f}")
            print(f"New prompt norm: {new_norm.item():.6f}")
            print(f"Prompt norm change: {diff_norm.item():.6f}")

            # Reset gradients and step the scheduler
            optimizer.zero_grad()
            gradient_accumulation_step = 0
            scheduler.step()
        else:
            gradient_accumulation_step += 1

        # Uncomment and adjust probability to occasionally run generation tests
        # if random.random() < 0.001:  # Rarely run generation tests during training
        #     _test_generation(model, processor, inputs_question, question, message, img_dir_q)

    # Save model checkpoint at the end of the epoch
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)
    state_dict = model.save_vip()
    torch.save(state_dict, os.path.join(save_base_path, f'vip_token.pt'))


def _test_generation(model, processor, inputs_question, question, message, img_dir):
    """
    Helper function to test model generation during training (for debugging)
    """
    with torch.no_grad():
        model.eval()

        cprint(f"Image: {img_dir}", 'yellow')
        print("=" * 50)
        print("Question:")
        cprint(question, 'blue')
        print("=" * 50)
        print("Model answer:")

        # Generate answer
        generated_ids = model.generate(**inputs_question, max_new_tokens=DEFAULT_CONFIG['MAX_NEW_TOKENS'])
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_question.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        cprint(output_text[0], "green")
        print("=" * 50)
        print("Ground truth answer:")
        cprint(message, "red")
        print("=" * 50)


def parse_arguments():
    """
    Parse command-line arguments for training configuration

    Returns:
        argparse.Namespace: parsed arguments containing training settings
    """
    parser = argparse.ArgumentParser(
        description="Train VIP DFD LLM model for deepfake detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and training configuration
    parser.add_argument('--train_json_path', type=str, default='./FaceDATA/stage2_id0_only_cls.json',
                        help='Path to training dataset JSON file')
    parser.add_argument('--name', type=str, default='Amair',
                        help='VIP user name')
    parser.add_argument('--device', type=str, default='3',
                        help='CUDA device ID to use for training')
    parser.add_argument('--optim_facechecker', action='store_true',
                        help='Enable optimization of the face checker')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Enable mixed precision training to save memory and speed up training')
    parser.add_argument('--mixed_precision_dtype', type=str, default='fp16', 
                        choices=['fp16', 'bf16'],
                        help='Data type for mixed precision training: fp16 or bf16 (default: fp16)')
    parser.add_argument('--epoch', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--token_num', type=int, default=32,
                        help='Number of VIP tokens to use')
    parser.add_argument('--gradient_accumulation_step', type=int, default=8,
                        help='Number of steps to accumulate gradients before updating')

    return parser.parse_args()


def main():
    """
    Main training function: initialize model, dataset and training loop
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # Define checkpoint save path
    save_base_path = f'./checkpoints/Stage3/{args.name}'

    # Check whether the model has already been trained
    checkpoint_path = os.path.join(save_base_path, f'vip_token.pt')
    if os.path.exists(checkpoint_path):
        print(f"Model checkpoint already exists at: {checkpoint_path}")
        print("Skipping training.")
        return

    # Import required modules (import here to avoid errors if dependencies are missing)
    try:
        from VIP_Dataset import VIP_Dataset
        from Models.VIPGuard import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        from Models.Wrapper import Wrapper
    except ImportError as e:
        print(f"Failed to import required modules: {e}")
        print("Please ensure all dependencies are installed and paths are correct.")
        return

    # Load pretrained model and processor
    pretrain_path = DEFAULT_CONFIG['PRETRAIN_PATH']
    print(f"Loading pretrained model from: {pretrain_path}")

    try:
        vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrain_path, 
            device_map='auto'
        )
        processor = Qwen2_5_VLProcessor.from_pretrained(
            pretrain_path, 
            device_map='auto'
        )
    except Exception as e:
        print(f"Failed to load pretrained model: {e}")
        return

    # Create model wrapper
    model = Wrapper(
        vl_model=vl_model, 
        processor=processor, 
        optim_facechecker=args.optim_facechecker,
        vip_token_num=args.token_num
    )

    # Load training dataset
    train_json_path = args.train_json_path
    print(f"Loading training data from: {train_json_path}")

    try:
        train_set = VIP_Dataset(train_json_path, processor=processor)
        train_loader = DataLoader(
            train_set, 
            batch_size=1, 
            num_workers=1, 
            pin_memory=True, 
            shuffle=True
        )
    except Exception as e:
        print(f"Failed to load training dataset: {e}")
        return

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.get_optim_params(), 
        lr=DEFAULT_CONFIG['DEFAULT_LR'],
        weight_decay=DEFAULT_CONFIG['WEIGHT_DECAY']
    )

    # Compute total number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params:,}")

    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader) // args.gradient_accumulation_step,
        eta_min=optimizer.param_groups[0]['lr'] * 0.0001
    )

    # Training loop
    print(f"Starting training for {args.epoch} epochs")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_step}")
    print(f"Total training batches: {len(train_loader)}")
    if args.use_mixed_precision:
        print(f"Mixed precision training: Enabled ({args.mixed_precision_dtype.upper()})")
    else:
        print("Mixed precision training: Disabled (FP32)")

    for epoch in range(args.epoch):
        print(f"\n{'='*60}")
        print(f"Training Epoch {epoch + 1}/{args.epoch}")
        print(f"{'='*60}")

        train_one_epoch(
            model=model,
            processor=processor,
            train_loader=train_loader,
            optimizer=optimizer,
            args=args,
            epoch=epoch + 1,
            scheduler=scheduler,
            save_base_path=save_base_path
        )

    print(f"\nTraining complete! Model saved to: {save_base_path}")


if __name__ == '__main__':
    main()
