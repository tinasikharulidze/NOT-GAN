# Standard libraries
import os
import sys
from collections import OrderedDict

# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image

from tqdm import tqdm 
# from tqdm.notebook import tqdm 

# PyTorch ecosystem
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Third-party models
import open_clip

# Your model definitions (needed because every function below instantiates a
# fresh Generator from model_path)
from models import ClipCondGeneratorFullyInjected


def plot_func(model_path, train_size, device, df, embeddings, my_seed, CHANNELS, IMG_SIZE, features=64):

    # 1. Initialize and load the Generator model (this block was missing --
    #    the function referenced `state_dict` and `G` without ever creating them)
    G = ClipCondGeneratorFullyInjected(channels_img=3, features_g=features).to(device)
    state_dict = torch.load(model_path, map_location=device)

    # 2. Strip "module." from the keys if it exists
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
        
    # 3. Load the cleaned state_dict
    G.load_state_dict(new_state_dict)
    G.eval()

    # ── Sample random indices ─────────────────────────────────────────────────────
    rng       = np.random.default_rng(42)
    train_idx = rng.choice(train_size,  size=10, replace=False)
    test_idx  = rng.choice(100_000 - train_size,   size=10, replace=False)

    # ── Get embeddings ────────────────────────────────────────────────────────────
    train_embs = torch.tensor(embeddings[df.iloc[train_idx]['emb_idx'].values], dtype=torch.float32, device=device)       # [10, 512]
    test_embs  = torch.tensor(embeddings[df.iloc[test_idx]['emb_idx'].values], dtype=torch.float32, device=device)        # [10, 512]
    null_embs  = G.null_token.detach().unsqueeze(0).expand(10, -1)  # [10, 512]

    # ── Fixed noise ───────────────────────────────────────────────────────────────
    torch.manual_seed(my_seed)
    noise_small = torch.rand(10, CHANNELS, 8, 8, device=device)
    noise       = F.interpolate(noise_small, size=(IMG_SIZE, IMG_SIZE),
                                mode='bilinear', align_corners=False)

    # ── Generate ──────────────────────────────────────────────────────────────────
    with torch.no_grad():
        out_train = G(noise, train_embs).cpu().clamp(0, 1)
        out_test  = G(noise, test_embs).cpu().clamp(0, 1)
        out_null  = G(noise, null_embs).cpu().clamp(0, 1)

    # ── Plot ──────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 10, figsize=(22, 7))
    row_labels = ['train emb', 'test emb', 'null token']

    for col in range(10):
        axes[0, col].imshow(out_train[col].permute(1, 2, 0).numpy())
        axes[1, col].imshow(out_test[col].permute(1, 2, 0).numpy())
        axes[2, col].imshow(out_null[col].permute(1, 2, 0).numpy())

        # Updated titles to show category and the unique row index/ID
        axes[0, col].set_title(f"train - ID {train_idx[col]}", fontsize=8)
        axes[1, col].set_title(f"test - ID {test_idx[col]}", fontsize=8)
        axes[2, col].set_title(f"null - ID {col}", fontsize=8)

        for row in range(3):
            axes[row, col].axis('off')

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)

    fig.suptitle('Inference for train / test / null', fontsize=11)
    fig.tight_layout()
    plt.show()

    # ── Print Descriptions ────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(" TRAIN CLIP TEXT DESCRIPTIONS")
    print("="*80)
    for idx in train_idx:
        desc = df.iloc[idx]['clip_text_description']
        print(f"ID {idx:5d}: {desc}")

    print("\n" + "="*80)
    print(" TEST CLIP TEXT DESCRIPTIONS")
    print("="*80)
    for idx in test_idx:
        desc = df.iloc[idx]['clip_text_description']
        print(f"ID {idx:5d}: {desc}")

def inference(text_prompts_list, model_path, device, clip_model, tokenizer, CHANNELS, IMG_SIZE, features_g=64):
    """
    Takes a list of text prompts, embeds them using a finetuned CLIP model,
    and generates 3 completely unique variations for each prompt (unique noise per image). 
    Plots them in a cohesive grid and prints descriptions by assigned ID.
    """
    if isinstance(text_prompts_list, str):
        text_prompts_list = [text_prompts_list]
        
    num_prompts = len(text_prompts_list)
    total_images = num_prompts * 3  # 3 variations per prompt

    # 1. Initialize and load your Generator model
    G = ClipCondGeneratorFullyInjected(channels_img=3, features_g=features_g).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    G.load_state_dict(new_state_dict)
    G.eval()

    clip_model.eval()

    # 3. Extract Finetuned CLIP text features for all prompts
    all_text_features = []
    with torch.no_grad():
        for prompt in text_prompts_list:
            tokenized_text = tokenizer([prompt]).to(device)
            text_features = clip_model.encode_text(tokenized_text)
            text_features /= text_features.norm(dim=-1, keepdim=True) 
            text_features = text_features.float() 
            
            # Replicate each feature mapping three times (for Var 1, Var 2, and Var 3)
            all_text_features.append(text_features.repeat(3, 1))
            
        # Shape: [num_prompts * 3, 512]
        batched_text_features = torch.cat(all_text_features, dim=0)

    # 4. Generate COMPLETELY UNIQUE noise for every single image slot
    # We remove the manual seed here so it's different every time you call it,
    # or you can leave it if you want the "randomness" to be reproducible.
    # torch.manual_seed(0) 
    
    # Generate a fresh noise batch of size [total_images, CHANNELS, 8, 8]
    noise_small = torch.rand(total_images, CHANNELS, 8, 8, device=device)
    
    # Match spatial interpolation geometry
    noise = F.interpolate(noise_small, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)

    # 5. Run Generator Inference
    with torch.no_grad():
        generated_images = G(noise, batched_text_features).cpu().clamp(0, 1)

    # 6. Display the images in a grid format (3 rows for the 3 variations x N columns for prompts)
    fig, axes = plt.subplots(3, num_prompts, figsize=(2.2 * num_prompts, 7))
    
    # Standardize axes to 2D array format even if only 1 prompt is passed
    if num_prompts == 1:
        axes = axes.reshape(3, 1)

    for p_idx in range(num_prompts):
        # The batch tensor is packed as: [P1_V1, P1_V2, P1_V3, P2_V1, P2_V2, P2_V3...]
        img_idx_v1 = p_idx * 3
        img_idx_v2 = p_idx * 3 + 1
        img_idx_v3 = p_idx * 3 + 2
        
        # Variation 1 (Row 0)
        img_np1 = generated_images[img_idx_v1].permute(1, 2, 0).numpy()
        axes[0, p_idx].imshow(img_np1)
        axes[0, p_idx].axis('off')
        axes[0, p_idx].set_title(f"ID {p_idx + 1} - Image 1", fontsize=8)
        
        # Variation 2 (Row 1)
        img_np2 = generated_images[img_idx_v2].permute(1, 2, 0).numpy()
        axes[1, p_idx].imshow(img_np2)
        axes[1, p_idx].axis('off')
        axes[1, p_idx].set_title(f"ID {p_idx + 1} - Image 2", fontsize=8)

        # Variation 3 (Row 2)
        img_np3 = generated_images[img_idx_v3].permute(1, 2, 0).numpy()
        axes[2, p_idx].imshow(img_np3)
        axes[2, p_idx].axis('off')
        axes[2, p_idx].set_title(f"ID {p_idx + 1} - Image 3", fontsize=8)

    axes[0, 0].set_ylabel("Variation 1", fontsize=9)
    axes[1, 0].set_ylabel("Variation 2", fontsize=9)
    axes[2, 0].set_ylabel("Variation 3", fontsize=9)
    
    fig.suptitle('Inference based on custom prompts', fontsize=11, weight='bold')
    fig.tight_layout()
    plt.show()

    # ── Print Descriptions ────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(" GENERATED PROMPT DESCRIPTIONS")
    print("="*80)
    for i, prompt in enumerate(text_prompts_list):
        print(f"ID {i + 1}: {prompt}")
    print("="*80)


def analyze_noise_schedules(test_prompt, model_path, device, clip_model, tokenizer, CHANNELS, IMG_SIZE, image_path,
                             features_g=64):
    """
    Takes a single text prompt, embeds it, and evaluates 6 different initial noise schedules.
    Uses a fixed real image from a path at FULL resolution to preserve clarity.
    Plots a 2x6 grid with thin black borders framing every single image asset.
    """
    # 1. Initialize and load your Generator model
    G = ClipCondGeneratorFullyInjected(channels_img=3, features_g=features_g).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    G.load_state_dict(new_state_dict)
    G.eval()
    
    clip_model.eval()

    # 2. Extract and Normalize CLIP text embedding for the prompt
    with torch.no_grad():
        tokenized_text = tokenizer([test_prompt]).to(device)
        text_features = clip_model.encode_text(tokenized_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()  # Shape: [1, 512]

    # 3. Load and prepare the specific real image from path at full resolution
    pil_img = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    real_img_tensor = transform(pil_img).to(device).unsqueeze(0) # Shape: [1, 3, IMG_SIZE, IMG_SIZE]

    # 4. Construct the 5 Synthetic Initial Noise Schedules at the 8x8 base footprint
    schedules_small = {}
    schedules_small["All Black"] = torch.zeros(1, CHANNELS, 8, 8, device=device)
    schedules_small["All White"] = torch.ones(1, CHANNELS, 8, 8, device=device)
    schedules_small["Uniform [0,1]"] = torch.rand(1, CHANNELS, 8, 8, device=device)
    schedules_small["Gaussian (St Dev = 0.1)"] = torch.randn(1, CHANNELS, 8, 8, device=device) * 0.1 + 0.5
    schedules_small["Gaussian (St Dev = 2)"] = torch.randn(1, CHANNELS, 8, 8, device=device) * 2.0 + 0.5

    # Interpolate small schedules up to the generator's resolution footprint
    final_inputs = {}
    for name, small_tensor in schedules_small.items():
        final_inputs[name] = F.interpolate(small_tensor, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    
    # Keep the original image perfectly crisp at full resolution instead of downsampling it to 8x8
    final_inputs["Initial Real Img"] = real_img_tensor

    # Ordered sequence layout
    schedule_order = [
        "All Black", 
        "All White", 
        "Initial Real Img", 
        "Uniform [0,1]", 
        "Gaussian (St Dev = 0.1)", 
        "Gaussian (St Dev = 2)"
    ]

    # 5. Run Generator Inference Across Lineup
    generated_outputs = {}
    batched_prompt = text_features.repeat(1, 1) # [1, 512]
    
    with torch.no_grad():
        for name in schedule_order:
            inp_noise = final_inputs[name]
            generated_outputs[name] = G(inp_noise, batched_prompt).cpu().clamp(0, 1).squeeze(0)

    # 6. Coordinate Grid Layout Construction (2 Rows x 6 Columns)
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    
    for col, name in enumerate(schedule_order):
        # --- ROW 1: INPUT ILLUSTRATION ---
        inp_img = final_inputs[name].cpu().squeeze(0).permute(1, 2, 0).numpy()
        
        # Smart visibility mapping for visualization sanity
        img_min, img_max = inp_img.min(), inp_img.max()
        if np.allclose(img_min, img_max):
            inp_img = np.clip(inp_img, 0.0, 1.0)
        else:
            inp_img = (inp_img - img_min) / (img_max - img_min + 1e-8)
        
        axes[0, col].imshow(inp_img)
        axes[0, col].set_title(name, fontsize=9, weight='bold')
        axes[0, col].axis('off')
        
        # Draw a thin black frame around the input image subplot boundaries
        rect_inp = patches.Rectangle((0, 0), inp_img.shape[1]-1, inp_img.shape[0]-1, 
                                     linewidth=1, edgecolor='black', facecolor='none')
        axes[0, col].add_patch(rect_inp)
        
        # --- ROW 2: OUTPUT ILLUSTRATION ---
        out_img = generated_outputs[name].permute(1, 2, 0).numpy()
        axes[1, col].imshow(out_img)
        axes[1, col].axis('off')
        
        # Draw a thin black frame around the output image subplot boundaries
        rect_out = patches.Rectangle((0, 0), out_img.shape[1]-1, out_img.shape[0]-1, 
                                     linewidth=1, edgecolor='black', facecolor='none')
        axes[1, col].add_patch(rect_out)

    # Row metadata labels
    axes[0, 0].set_ylabel("INPUT (Noise/Img)", fontsize=11, weight='bold', labelpad=15)
    axes[1, 0].set_ylabel("OUTPUT (Generated)", fontsize=11, weight='bold', labelpad=15)
    
    # Restore specific axes parameters to prevent label cropping
    axes[0, 0].axis('on')
    axes[1, 0].axis('on')
    axes[0, 0].set_xticks([]), axes[0, 0].set_yticks([])
    axes[1, 0].set_xticks([]), axes[1, 0].set_yticks([])

    fig.suptitle(f"Noise Schedule Analysis\nPrompt: \"{test_prompt}\"", 
                 fontsize=12, weight='bold', y=1.02)
    fig.tight_layout()
    plt.show()


def compare_train_test_reconstruction(model_path, device, clip_model, tokenizer, CHANNELS, IMG_SIZE,
                                       train_path,
                                       train_desc,
                                       test_path,
                                       test_desc, features_g=64):
    """
    Plots a 2x2 grid comparing a specific training and testing sample:
    - Column 1: Real Train Image vs. Model Generation from Train Text
    - Column 2: Real Test Image vs. Model Generation from Test Text
    Real images are kept in raw 0-1 scale without any normalisation.
    Every image is enclosed in a thin black frame.
    """

    # 2. Initialize and load your Generator model
    G = ClipCondGeneratorFullyInjected(channels_img=3, features_g=features_g).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    G.load_state_dict(new_state_dict)
    G.eval()
    
    clip_model.eval()

    # 3. Extract and Normalize CLIP text embeddings for both descriptions
    with torch.no_grad():
        # Train Embedding
        tokenized_train = tokenizer([train_desc]).to(device)
        train_features = clip_model.encode_text(tokenized_train)
        train_features /= train_features.norm(dim=-1, keepdim=True)
        
        # Test Embedding
        tokenized_test = tokenizer([test_desc]).to(device)
        test_features = clip_model.encode_text(tokenized_test)
        test_features /= test_features.norm(dim=-1, keepdim=True)
        
        # Batch them together: Shape [2, 512]
        batched_text_features = torch.cat([train_features, test_features], dim=0).float()

    # 4. Load Real Images strictly in [0, 1] range without standard normalization
    pil_train = Image.open(train_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    pil_test = Image.open(test_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    
    # Simple division to scale to 0-1 range
    real_train_np = np.array(pil_train, dtype=np.float32) / 255.0
    real_test_np = np.array(pil_test, dtype=np.float32) / 255.0

    # 5. Generate reproducible underlying noise layouts using StDev 1
    torch.manual_seed(0) 
    noise_small = torch.randn(2, CHANNELS, 8, 8, device=device) * 1.0  # StDev 1 base layout
    noise = F.interpolate(noise_small, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)

    # 6. Run Generator Inference
    with torch.no_grad():
        generated_images = G(noise, batched_text_features).cpu().clamp(0, 1)
        
    out_train_np = generated_images[0].permute(1, 2, 0).numpy()
    out_test_np = generated_images[1].permute(1, 2, 0).numpy()

    # 7. Build the 2x2 Plot Grid
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))
    
    # --- COLUMN 1: TRAINING DATA ---
    # Row 1: Real Train Image
    axes[0, 0].imshow(real_train_np)
    axes[0, 0].set_title("Real Train Image", fontsize=10, weight='bold')
    axes[0, 0].axis('off')
    rect1 = patches.Rectangle((0, 0), IMG_SIZE-1, IMG_SIZE-1, linewidth=1, edgecolor='black', facecolor='none')
    axes[0, 0].add_patch(rect1)
    
    # Row 2: Generated Train Image
    axes[1, 0].imshow(out_train_np)
    axes[1, 0].set_title("Generated (Train Prompt)", fontsize=10, weight='bold')
    axes[1, 0].axis('off')
    rect2 = patches.Rectangle((0, 0), IMG_SIZE-1, IMG_SIZE-1, linewidth=1, edgecolor='black', facecolor='none')
    axes[1, 0].add_patch(rect2)

    # --- COLUMN 2: TESTING DATA ---
    # Row 1: Real Test Image
    axes[0, 1].imshow(real_test_np)
    axes[0, 1].set_title("Real Test Image", fontsize=10, weight='bold')
    axes[0, 1].axis('off')
    rect3 = patches.Rectangle((0, 0), IMG_SIZE-1, IMG_SIZE-1, linewidth=1, edgecolor='black', facecolor='none')
    axes[0, 1].add_patch(rect3)
    
    # Row 2: Generated Test Image
    axes[1, 1].imshow(out_test_np)
    axes[1, 1].set_title("Generated (Test Prompt)", fontsize=10, weight='bold')
    axes[1, 1].axis('off')
    rect4 = patches.Rectangle((0, 0), IMG_SIZE-1, IMG_SIZE-1, linewidth=1, edgecolor='black', facecolor='none')
    axes[1, 1].add_patch(rect4)

    # Global formatting labels
    axes[0, 0].axis('on')
    axes[1, 0].axis('on')
    axes[0, 0].set_ylabel("REAL TARGET", fontsize=11, weight='bold', labelpad=15)
    axes[1, 0].set_ylabel("MODEL OUTPUT", fontsize=11, weight='bold', labelpad=15)
    axes[0, 0].set_xticks([]), axes[0, 0].set_yticks([])
    axes[1, 0].set_xticks([]), axes[1, 0].set_yticks([])

    fig.suptitle("Paired Reconstruction Analysis: Train vs. Test", fontsize=12, weight='bold', y=0.98)
    fig.tight_layout()
    plt.show()

    print(train_desc)
    print("="*80)
    print(test_desc)


def run_dataset_average_distance_test(df, embeddings, model_path, device, features_g=64):
    """
    Computes global averaged latent distances across the entire Train (first 80k) 
    and Test (last 1k) partitions of the dataframe.
    """
    # --- 1. Load Generator to extract the learned Null Token baseline ---
    G = ClipCondGeneratorFullyInjected(channels_img=3, features_g=features_g).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    G.load_state_dict(new_state_dict)
    G.eval()
    
    # Extract and normalize the Null Token embedding [512]
    emb_null = G.null_token.detach().unsqueeze(0)
    emb_null_norm = emb_null / emb_null.norm(dim=-1, keepdim=True)
    v_null = emb_null_norm.cpu().float().numpy().flatten()

    # --- 2. Separate Train vs Test Indices ---
    # Using your `emb_idx` column to pull from your precomputed `embeddings` matrix
    train_df = df.iloc[:80000]
    test_df = df.iloc[-1000:]
    
    train_indices = train_df['emb_idx'].values
    test_indices = test_df['emb_idx'].values
    
    # Pull arrays and ensure they are unit-normalized
    # If your 'embeddings' bank isn't already normalized, uncomment the lines below:
    # embs_train = embeddings[train_indices] / np.linalg.norm(embeddings[train_indices], axis=1, keepdims=True)
    # embs_test = embeddings[test_indices] / np.linalg.norm(embeddings[test_indices], axis=1, keepdims=True)
    embs_train = embeddings[train_indices]
    embs_test = embeddings[test_indices]

    # --- 3. Compute Metrics ---
    
    # A. Pairwise Distances: Test vs. Global Null Token Baseline
    # Dot product of normalized matrices yields cosine similarity matrix
    cos_sim_test_null = np.dot(embs_test, v_null) 
    l2_dist_test_null = np.linalg.norm(embs_test - v_null, axis=1)
    
    # B. Pairwise Distances: Train vs. Global Null Token Baseline
    cos_sim_train_null = np.dot(embs_train, v_null)
    l2_dist_train_null = np.linalg.norm(embs_train - v_null, axis=1)

    # C. Cross-Dataset Distance: Test Prompts vs. Closest Training Concepts
    # For every test prompt, we find the single training prompt it is closest to 
    # to measure how "out-of-distribution" the test set actually is.
    max_cos_sim_test_train = []
    min_l2_dist_test_train = []
    
    # Process in batches to keep memory usage low
    batch_size = 100
    for i in range(0, len(embs_test), batch_size):
        batch_test = embs_test[i:i+batch_size]
        
        # Matrix multiply: [batch_size, 512] x [512, 80000] -> [batch_size, 80000]
        similarity_matrix = np.dot(batch_test, embs_train.T)
        
        # Best matches per test sample in this batch
        max_sims = np.max(similarity_matrix, axis=1)
        max_cos_sim_test_train.extend(max_sims)
        
        # For L2 distance: standard formula sqrt(2 - 2 * cos_sim) for normalized vectors
        min_l2s = torch.sqrt(torch.clamp(2.0 - 2.0 * torch.tensor(max_sims), min=0.0)).numpy()
        min_l2_dist_test_train.extend(min_l2s)

    # --- 4. Report Global Averages ---
    print("=" * 70)
    print("      DATASET-WIDE AVERAGED DISTANCE ANALYSIS")
    print("=" * 70)
    print(f"1. [TEST SET] (CLOSEST MATCH IN TRAIN SET) (Average out of 1,000 samples)")
    print(f"   - Mean Best Cosine Similarity:  {np.mean(max_cos_sim_test_train):.4f}")
    print(f"   - Mean Nearest Neighbor L2 Dist: {np.mean(min_l2_dist_test_train):.4f}")
    print("-" * 70)
    print(f"2. [TEST SET] (NULL TOKEN BASELINE) (Average out of 1,000 samples)")
    print(f"   - Mean Cosine Similarity:       {np.mean(cos_sim_test_null):.4f}")
    print(f"   - Mean L2 Distance:             {np.mean(l2_dist_test_null):.4f}")
    print("-" * 70)
    print(f"3. [TRAIN SET] (NULL TOKEN GLOBAL BASELINE) (Average out of 80,000 samples)")
    print(f"   - Mean Cosine Similarity:       {np.mean(cos_sim_train_null):.4f}")
    print(f"   - Mean L2 Distance:             {np.mean(l2_dist_train_null):.4f}")
    print("=" * 70)
    
    # --- 5. Quick Quantitative Interpretation ---
    test_null_dist = np.mean(l2_dist_test_null)
    train_null_dist = np.mean(l2_dist_train_null)



def plot_comparative_clip_distributions_with_random(df, embeddings, model_path, device, clip_model, CHANNELS, IMG_SIZE, features_g=64):
    """
    Computes and plots global average CLIP text-to-image similarity scores for:
    1. Training set (Paired)
    2. Testing set (Paired)
    3. Testing set (Unpaired/Permuted Random Baseline)
    """
    # 1. Initialize and load your Generator model
    G = ClipCondGeneratorFullyInjected(channels_img=3, features_g=features_g).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    G.load_state_dict(new_state_dict)
    G.eval()
    
    clip_model.eval()

    # 2. Extract Partition Indices from the DataFrame
    train_df_sample = df.iloc[:1000]
    test_df = df.iloc[-1000:]
    
    train_indices = train_df_sample['emb_idx'].values
    test_indices = test_df['emb_idx'].values
    
    # Extract precomputed text embeddings matrices: Shape [1000, 512] each
    text_embs_train = embeddings[train_indices]
    text_embs_test = embeddings[test_indices]
    
    # Convert to PyTorch tensors
    text_embs_train_tensor = torch.tensor(text_embs_train, dtype=torch.float32, device=device)
    text_embs_test_tensor = torch.tensor(text_embs_test, dtype=torch.float32, device=device)

    # 3. Standard CLIP Image Preprocessing Pipeline
    clip_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.48145466, 0.45782711, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # 4. Helper Function to batch process and extract generated image features
    def get_generated_image_embeddings(text_features_tensor, partition_label):
        batch_size = 50
        all_image_features = []
        
        print(f"Processing {partition_label} partition...")
        for i in tqdm(range(0, len(text_features_tensor), batch_size)):
            batch_text = text_features_tensor[i:i+batch_size]
            curr_batch_size = batch_text.size(0)

            with torch.no_grad():
                torch.manual_seed(i) 
                noise_small = torch.randn(curr_batch_size, CHANNELS, 8, 8, device=device) * 1.0
                noise = F.interpolate(noise_small, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)

                # Execute forward pass through Generator
                gen_images = G(noise, batch_text).clamp(0, 1)
                processed_images = clip_preprocess(gen_images)

                # Extract unit-normalized visual embeddings from CLIP
                img_feats = clip_model.encode_image(processed_images)
                img_feats /= img_feats.norm(dim=-1, keepdim=True)
                
                all_image_features.append(img_feats.cpu())
                
        return torch.cat(all_image_features, dim=0).numpy()

    # 5. Extract Visual Features for Both Sets
    gen_image_embs_train = get_generated_image_embeddings(text_embs_train_tensor, "TRAINING")
    gen_image_embs_test = get_generated_image_embeddings(text_embs_test_tensor, "TESTING")

    # 6. Compute True Pairwise Row-by-Row Cosine Similarities
    train_scores = np.sum(text_embs_train * gen_image_embs_train, axis=1)
    test_scores = np.sum(text_embs_test * gen_image_embs_test, axis=1)

    # 7. Compute Permuted Random Baseline Score
    # np.roll shifts the array elements by 1 position (element 0 pairs with element 1, etc.)
    # This guarantees that every text embedding is matched with a completely unrelated image
    permuted_text_embs_test = np.roll(text_embs_test, shift=1, axis=0)
    random_scores = np.sum(permuted_text_embs_test * gen_image_embs_test, axis=1)

    # Compute Global Means
    mean_train = np.mean(train_scores)
    mean_test = np.mean(test_scores)
    mean_random = np.mean(random_scores)

    # 8. Generate Distribution Plots
    plt.figure(figsize=(11, 6.5))
    
    # Plot Training Distribution
    sns.histplot(train_scores, kde=True, color='royalblue', alpha=0.35, stat="density", 
                 label=f'Train Partition (Mean: {mean_train:.4f})', bins=30)
    
    # Plot Testing Distribution
    sns.histplot(test_scores, kde=True, color='crimson', alpha=0.35, stat="density", 
                 label=f'Test Partition (Mean: {mean_test:.4f})', bins=30)
    
    # Plot Permuted Random Baseline Distribution
    sns.histplot(random_scores, kde=True, color='gray', alpha=0.3, stat="density", 
                 label=f'Random Unrelated Baseline (Mean: {mean_random:.4f})', bins=30)
    
    # Add vertical indicators for the global means
    plt.axvline(mean_train, color='darkblue', linestyle='--', linewidth=1.5)
    plt.axvline(mean_test, color='darkred', linestyle='--', linewidth=1.5)
    plt.axvline(mean_random, color='black', linestyle=':', linewidth=1.5)

    # Format styling labels
    plt.title("Cross-Partition vs. Random Permuted CLIP Alignment Distribution", fontsize=13, weight='bold', pad=15)
    plt.xlabel("Cosine Similarity Score", fontsize=11, labelpad=10)
    plt.ylabel("Density / Probability", fontsize=11, labelpad=10)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.show()


def plot_latent_guidance(prompt, w_values, G, clip_model, tokenizer, device):
    """
    Sweeps through guidance scales (w) to visualize latent interpolation and extrapolation.
    Generates 2 rows using different structural noise seeds to show consistency.
    """
    # Ensure models are in evaluation mode
    G.eval()
    clip_model.eval()

    with torch.no_grad():
        # 1. Extract Class Embedding (Text)
        tokens = tokenizer([prompt]).to(device)
        class_emb = clip_model.encode_text(tokens)
        class_emb = F.normalize(class_emb, dim=-1).float()  # Shape: [1, 512]
        
        # 2. Extract Null Embedding (Unconditional Base)
        null_emb = G.null_token.detach().unsqueeze(0)       # Shape: [1, 512]

        # 3. Generate TWO different fixed noise layouts (Batch size = 2)
        torch.manual_seed(42)
        noise_small_1 = torch.rand(1, 3, 8, 8, device=device)
        
        torch.manual_seed(999) # Second distinct layout
        noise_small_2 = torch.rand(1, 3, 8, 8, device=device)
        
        # Combine them into a single batch and upsample
        noise_small_batch = torch.cat([noise_small_1, noise_small_2], dim=0) # Shape: [2, 3, 8, 8]
        noise = F.interpolate(noise_small_batch, size=(64, 64), mode='bilinear', align_corners=False)

        # 4. Generate across different W weights
        row1_images = []
        row2_images = []
        
        for w in w_values:
            # LATENT GUIDANCE MATH
            guided_emb = null_emb + w * (class_emb - null_emb)
            
            # CRITICAL FIX: Pull the vector back onto the normalized hypersphere
            guided_emb = F.normalize(guided_emb, dim=-1).float()
            
            # Duplicate the embedding so it matches our noise batch size of 2
            guided_emb_batched = guided_emb.repeat(2, 1)
            
            # Generate both images simultaneously
            img_batch = G(noise, guided_emb_batched).cpu().clamp(0, 1)
            
            # Separate the batch into our two rows
            row1_images.append(img_batch[0].permute(1, 2, 0).detach().numpy())
            row2_images.append(img_batch[1].permute(1, 2, 0).detach().numpy())

    # 5. Plotting
    fig, axes = plt.subplots(2, len(w_values), figsize=(16, 7))
    
    for idx, w in enumerate(w_values):
        # Plot Row 1
        axes[0, idx].imshow(row1_images[idx])
        axes[0, idx].axis('off')
        
        # Plot Row 2
        axes[1, idx].imshow(row2_images[idx])
        axes[1, idx].axis('off')
        
        # Add descriptive titles to the top row only
        if w == 0:
            title = f"w = {w}\n(Null / Unconditional)"
        elif w == 1:
            title = f"w = {w}\n(Standard Model)"
        else:
            title = f"w = {w}\n(Interpolated/Extrapolated)"
            
        axes[0, idx].set_title(title, fontsize=11, linespacing=1.5)

    # Add row labels for clarity
    axes[0, 0].set_ylabel("Seed 42", fontsize=12, labelpad=10)
    axes[0, 0].axis('on') # Turn axis back on just to show the ylabel
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    for spine in axes[0, 0].spines.values(): spine.set_visible(False)

    axes[1, 0].set_ylabel("Seed 999", fontsize=12, labelpad=10)
    axes[1, 0].axis('on')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    for spine in axes[1, 0].spines.values(): spine.set_visible(False)

    fig.suptitle(f"Classifier-Free Guidance (CFG)\nPrompt: '{prompt}'", fontsize=14, weight='bold', y=1.05)
    plt.tight_layout()
    plt.show()


def plot_latent_noise_optimization(
    prompt, 
    w_values, 
    G, 
    D, 
    clip_model, 
    tokenizer, 
    device,
    steps=40, 
    opt_lr=0.03, 
):
    """
    Finds the noise vector z that maximizes prompt-specificity using the Discriminator as a judge.
    Sweeps through multiple w_values across 2 different initial noise configurations.
    """
    # 1. Set models to correct modes
    G.eval()  # Keep G frozen (we only train z)
    D.eval()  # Keep D frozen
    clip_model.eval()

    # 2. Extract Text & Null Embeddings (No gradients needed here)
    with torch.no_grad():
        tokens = tokenizer([prompt]).to(device)
        class_emb = clip_model.encode_text(tokens)
        class_emb = F.normalize(class_emb, dim=-1).float()  # [1, 512]
        null_emb = G.null_token.detach().unsqueeze(0)       # [1, 512]

    row1_images = []
    row2_images = []

    # 3. Sweep across each guidance weight
    for w in w_values:
        print(f"Optimizing latent noise for w = {w}...")
        
        # Define base noise states for both rows
        torch.manual_seed(42)
        base_noise_1 = torch.rand(1, 3, 8, 8, device=device)
        
        torch.manual_seed(999)
        base_noise_2 = torch.rand(1, 3, 8, 8, device=device)
        
        # Combine into a batch [2, 3, 8, 8] and upscale to [2, 3, 64, 64]
        init_noise_batch = torch.cat([base_noise_1, base_noise_2], dim=0)
        init_noise_upscaled = F.interpolate(init_noise_batch, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Prepare the target variable z to receive gradients
        z = init_noise_upscaled.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z], lr=opt_lr)
        
        # Prepare matched embedding shapes for the batch size of 2
        class_emb_batched = class_emb.repeat(2, 1)
        null_emb_batched = null_emb.repeat(2, 1)

        # 4. The Optimization Loop
        for step in range(steps):
            # Generate images from current z state
            fake_cond   = G(z, class_emb_batched)
            fake_uncond = G(z, null_emb_batched)
            
            # Score images using your conditional Discriminator
            score_cond   = D(fake_cond,   class_emb_batched)
            score_uncond = D(fake_uncond, null_emb_batched)
            
            # Calculate Option B guidance equation
            guided_score = (1 + w) * score_cond - w * score_uncond
            
            # We want to MAXIMIZE the score, so we MINIMIZE the negative score
            loss = -guided_score.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Optional: Clamp z or keep it bounded if your GAN expects standard uniform/normal noise
            with torch.no_grad():
                z.clamp_(0.0, 1.0)

        # 5. Extract final optimized outputs
        with torch.no_grad():
            final_img_batch = G(z, class_emb_batched).cpu().clamp(0, 1)
            
            row1_images.append(final_img_batch[0].permute(1, 2, 0).numpy())
            row2_images.append(final_img_batch[1].permute(1, 2, 0).numpy())

    # 6. Plotting Results
    fig, axes = plt.subplots(2, len(w_values), figsize=(16, 7))
    
    for idx, w in enumerate(w_values):
        axes[0, idx].imshow(row1_images[idx])
        axes[0, idx].axis('off')
        
        axes[1, idx].imshow(row2_images[idx])
        axes[1, idx].axis('off')
        
        title = f"w = {w}\n(Optimized z)"
        axes[0, idx].set_title(title, fontsize=11, linespacing=1.5)

    # Clean styling for Row Labels
    axes[0, 0].set_ylabel("Seed 42 Base", fontsize=12, labelpad=10)
    axes[0, 0].axis('on')
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    for spine in axes[0, 0].spines.values(): spine.set_visible(False)

    axes[1, 0].set_ylabel("Seed 999 Base", fontsize=12, labelpad=10)
    axes[1, 0].axis('on')
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    for spine in axes[1, 0].spines.values(): spine.set_visible(False)

    fig.suptitle(f"Option B: Latent Noise Optimization via Discriminator\nPrompt: '{prompt}'", fontsize=14, weight='bold', y=1.05)
    plt.tight_layout()
    plt.show()



def slerp(v0, v1, t):
    """Spherical linear interpolation between two unit vectors."""
    v0 = F.normalize(v0, dim=-1)
    v1 = F.normalize(v1, dim=-1)
    dot = (v0 * v1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    
    # fallback to lerp when vectors are nearly parallel
    sin_omega = torch.sin(omega)
    safe = sin_omega.abs() > 1e-6
    
    out = torch.where(
        safe,
        (torch.sin((1 - t) * omega) / sin_omega) * v0 +
        (torch.sin(t * omega)       / sin_omega) * v1,
        (1 - t) * v0 + t * v1   # lerp fallback
    )
    return F.normalize(out, dim=-1)

def plot_slerp_guidance_experiment(prompt, w_values, G, clip_model, tokenizer, clip_preprocess, device):
    """
    Sweeps through guidance scales (w) using SLERP for latent extrapolation.
    Generates a visual grid and plots the CLIP cosine similarity scores.
    """
    G.eval()
    clip_model.eval()

    row1_images = []
    row2_images = []
    clip_scores = []

    with torch.no_grad():
        # 1. Extract and Normalize Embeddings
        tokens = tokenizer([prompt]).to(device)
        class_emb = clip_model.encode_text(tokens)
        class_emb = F.normalize(class_emb, dim=-1).float()  # [1, 512]
        
        null_emb = G.null_token.detach().unsqueeze(0)       # [1, 512]
        null_emb = F.normalize(null_emb, dim=-1).float()

        # 2. Generate TWO different fixed noise layouts
        torch.manual_seed(42)
        noise_small_1 = torch.rand(1, 3, 8, 8, device=device)
        
        torch.manual_seed(999) 
        noise_small_2 = torch.rand(1, 3, 8, 8, device=device)
        
        noise_small_batch = torch.cat([noise_small_1, noise_small_2], dim=0) 
        noise = F.interpolate(noise_small_batch, size=(64, 64), mode='bilinear', align_corners=False)

        # 3. Generate across different W weights
        for w in w_values:
            # LATENT GUIDANCE MATH: Now using SLERP instead of LERP
            guided_emb = slerp(null_emb, class_emb, w)
            
            # Duplicate the embedding so it matches our noise batch size of 2
            guided_emb_batched = guided_emb.repeat(2, 1)
            
            # Generate both images simultaneously
            img_batch = G(noise, guided_emb_batched).cpu().clamp(0, 1)
            
            # Save for plotting
            img1 = img_batch[0].permute(1, 2, 0).detach().numpy()
            img2 = img_batch[1].permute(1, 2, 0).detach().numpy()
            row1_images.append(img1)
            row2_images.append(img2)

            # 4. Calculate CLIP Score (Cosine Similarity) against Seed 42
            # Note: We use the PyTorch tensor of the image, not the numpy array
            img_for_clip = img_batch[0].unsqueeze(0) 
            img_feat = clip_model.encode_image(clip_preprocess(img_for_clip).to(device))
            img_feat = F.normalize(img_feat, dim=-1).float()
            
            score = (img_feat * class_emb).sum().item()
            clip_scores.append(score)

    # ─── 5. PLOTTING THE IMAGE GRID ──────────────────────────────────────────
    fig, axes = plt.subplots(2, len(w_values), figsize=(16, 7))
    
    for idx, w in enumerate(w_values):
        # Plot Row 1
        axes[0, idx].imshow(row1_images[idx])
        axes[0, idx].axis('off')
        
        # Plot Row 2
        axes[1, idx].imshow(row2_images[idx])
        axes[1, idx].axis('off')
        
        # Add descriptive titles
        if w == 0:
            title = f"w = {w}\n(Null / Unconditional)"
        elif w == 1:
            title = f"w = {w}\n(Standard Model)"
        else:
            title = f"w = {w}\n(SLERP Extrapolated)"
            
        axes[0, idx].set_title(title, fontsize=11, linespacing=1.5)

    # Add row labels
    axes[0, 0].set_ylabel("Seed 42", fontsize=12, labelpad=10)
    axes[0, 0].axis('on') 
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    for spine in axes[0, 0].spines.values(): spine.set_visible(False)

    axes[1, 0].set_ylabel("Seed 999", fontsize=12, labelpad=10)
    axes[1, 0].axis('on')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    for spine in axes[1, 0].spines.values(): spine.set_visible(False)

    fig.suptitle(f"SLERP Latent Guidance Experiment\nPrompt: '{prompt}'", fontsize=14, weight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

    # ─── 6. PLOTTING THE CLIP SCORE GRAPH ────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(w_values, clip_scores, marker='o', linestyle='-', color='b', linewidth=2)
    
    # Highlight the "Perfect" Text Prompt point (w=1)
    if 1.0 in w_values:
        idx_w1 = w_values.index(1.0)
        plt.scatter(1.0, clip_scores[idx_w1], color='red', s=100, zorder=5, label="w=1 (Text Prompt)")
    
    plt.title("Semantic Alignment (CLIP Cosine Similarity) vs. Guidance Weight (w)")
    plt.xlabel("Guidance Weight (w)")
    plt.ylabel("Cosine Similarity Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()



def slerp(v0, v1, t):
    """Spherical linear interpolation between two unit vectors."""
    v0 = F.normalize(v0, dim=-1)
    v1 = F.normalize(v1, dim=-1)
    dot = (v0 * v1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    
    # fallback to lerp when vectors are nearly parallel
    sin_omega = torch.sin(omega)
    safe = sin_omega.abs() > 1e-6
    
    out = torch.where(
        safe,
        (torch.sin((1 - t) * omega) / sin_omega) * v0 +
        (torch.sin(t * omega)       / sin_omega) * v1,
        (1 - t) * v0 + t * v1   # lerp fallback
    )
    return F.normalize(out, dim=-1)

def plot_latent_guidance_slerp(prompt, w_values, G, clip_model, tokenizer, device):
    """
    Sweeps through guidance scales (w) using SLERP for latent extrapolation.
    Generates 2 rows using different structural noise seeds to show consistency.
    """
    # Ensure models are in evaluation mode
    G.eval()
    clip_model.eval()

    with torch.no_grad():
        # 1. Extract Class Embedding (Text)
        tokens = tokenizer([prompt]).to(device)
        class_emb = clip_model.encode_text(tokens)
        class_emb = F.normalize(class_emb, dim=-1).float()  # Shape: [1, 512]
        
        # 2. Extract Null Embedding (Unconditional Base)
        null_emb = G.null_token.detach().unsqueeze(0)       # Shape: [1, 512]
        null_emb = F.normalize(null_emb, dim=-1).float()

        # 3. Generate TWO different fixed noise layouts (Batch size = 2)
        torch.manual_seed(42)
        noise_small_1 = torch.rand(1, 3, 8, 8, device=device)
        
        torch.manual_seed(1998) # Second distinct layout
        noise_small_2 = torch.rand(1, 3, 8, 8, device=device)
        
        # Combine them into a single batch and upsample
        noise_small_batch = torch.cat([noise_small_1, noise_small_2], dim=0) # Shape: [2, 3, 8, 8]
        noise = F.interpolate(noise_small_batch, size=(64, 64), mode='bilinear', align_corners=False)

        # 4. Generate across different W weights
        row1_images = []
        row2_images = []
        
        for w in w_values:
            # LATENT GUIDANCE MATH WITH SLERP
            guided_emb = slerp(null_emb, class_emb, w)
            
            # Duplicate the embedding so it matches our noise batch size of 2
            guided_emb_batched = guided_emb.repeat(2, 1)
            
            # Generate both images simultaneously
            img_batch = G(noise, guided_emb_batched).cpu().clamp(0, 1)
            
            # Separate the batch into our two rows
            row1_images.append(img_batch[0].permute(1, 2, 0).detach().numpy())
            row2_images.append(img_batch[1].permute(1, 2, 0).detach().numpy())

    # 5. Plotting
    fig, axes = plt.subplots(2, len(w_values), figsize=(16, 7))
    
    for idx, w in enumerate(w_values):
        # Plot Row 1
        axes[0, idx].imshow(row1_images[idx])
        axes[0, idx].axis('off')
        
        # Plot Row 2
        axes[1, idx].imshow(row2_images[idx])
        axes[1, idx].axis('off')
        
        # Add descriptive titles to the top row only
        if w == 0:
            title = f"w = {w}\n(Null / Unconditional)"
        elif w == 1:
            title = f"w = {w}\n(Standard Model)"
        else:
            title = f"w = {w}\n(SLERP Extrapolated)"
            
        axes[0, idx].set_title(title, fontsize=11, linespacing=1.5)

    # Add row labels for clarity
    axes[0, 0].set_ylabel("Seed 42", fontsize=12, labelpad=10)
    axes[0, 0].axis('on') # Turn axis back on just to show the ylabel
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    for spine in axes[0, 0].spines.values(): spine.set_visible(False)

    axes[1, 0].set_ylabel("Seed 1998", fontsize=12, labelpad=10)
    axes[1, 0].axis('on')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    for spine in axes[1, 0].spines.values(): spine.set_visible(False)

    fig.suptitle(f"Latent Space Interpolation with SLERP\nPrompt: '{prompt}'", fontsize=14, weight='bold', y=1.05)
    plt.tight_layout()
    plt.show()



def plot_text_interpolation(prompt_A, prompt_B, t_values, G, clip_model, tokenizer, device):
    """
    Sweeps through interpolation steps (t) between two text prompts using SLERP.
    t=0 perfectly matches prompt_A, t=1 perfectly matches prompt_B.
    """
    G.eval()
    clip_model.eval()

    with torch.no_grad():
        # 1. Extract Embeddings for BOTH Prompts
        tokens_A = tokenizer([prompt_A]).to(device)
        emb_A = F.normalize(clip_model.encode_text(tokens_A), dim=-1).float()
        
        tokens_B = tokenizer([prompt_B]).to(device)
        emb_B = F.normalize(clip_model.encode_text(tokens_B), dim=-1).float()

        # 2. Generate TWO different fixed noise layouts (Batch size = 2)
        torch.manual_seed(42)
        noise_small_1 = torch.rand(1, 3, 8, 8, device=device)
        
        torch.manual_seed(1998) 
        noise_small_2 = torch.rand(1, 3, 8, 8, device=device)
        
        noise_small_batch = torch.cat([noise_small_1, noise_small_2], dim=0) 
        noise = F.interpolate(noise_small_batch, size=(64, 64), mode='bilinear', align_corners=False)

        # 3. Generate across different 't' steps
        row1_images = []
        row2_images = []
        
        for t in t_values:
            # INTERPOLATION MATH: Walking the bridge from A to B
            guided_emb = slerp(emb_A, emb_B, t)
            
            # Duplicate the embedding so it matches our noise batch size of 2
            guided_emb_batched = guided_emb.repeat(2, 1)
            
            # Generate both images simultaneously
            img_batch = G(noise, guided_emb_batched).cpu().clamp(0, 1)
            
            # Separate the batch into our two rows
            row1_images.append(img_batch[0].permute(1, 2, 0).detach().numpy())
            row2_images.append(img_batch[1].permute(1, 2, 0).detach().numpy())

    # 4. Plotting
    fig, axes = plt.subplots(2, len(t_values), figsize=(2.5 * len(t_values), 7))
    
    for idx, t in enumerate(t_values):
        # Plot Row 1
        axes[0, idx].imshow(row1_images[idx])
        axes[0, idx].axis('off')
        
        # Plot Row 2
        axes[1, idx].imshow(row2_images[idx])
        axes[1, idx].axis('off')
        
        # Add descriptive titles to the top row
        if t == 0.0:
            title = f"t = {t}\n(100% Prompt A)"
        elif t == 1.0:
            title = f"t = {t}\n(100% Prompt B)"
        else:
            title = f"t = {t}\n(Blend)"
            
        axes[0, idx].set_title(title, fontsize=10, linespacing=1.5)

    # Add row labels for clarity
    axes[0, 0].set_ylabel("Seed 42", fontsize=12, labelpad=10)
    axes[0, 0].axis('on') 
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    for spine in axes[0, 0].spines.values(): spine.set_visible(False)

    axes[1, 0].set_ylabel("Seed 1998", fontsize=12, labelpad=10)
    axes[1, 0].axis('on')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    for spine in axes[1, 0].spines.values(): spine.set_visible(False)

    # Wrap the super title so it doesn't run off the screen
    wrapped_title = f"Latent Space Interpolation with SLERP\nPrompt A: '{prompt_A}'\nPrompt B: '{prompt_B}'"
    fig.suptitle(wrapped_title, fontsize=12, weight='bold', y=1.08)
    plt.tight_layout()
    plt.show()