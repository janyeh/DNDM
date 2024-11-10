import os
import glob

def check_model_checkpoints():
    checkpoint_patterns = [
        'output/netG_content_*.pth',
        'output/netG_haze_*.pth', 
        'output/net_dehaze_*.pth',
        'output/net_G_*.pth'
    ]

    print("\nChecking Model Checkpoints:")
    for pattern in checkpoint_patterns:
        files = glob.glob(pattern)
        print(f"\nLooking for {pattern}")
        if len(files) == 0:
            print(f"No checkpoints found")
            continue
            
        # Sort files by epoch number
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"Found {len(files)} checkpoints")
        print(f"Latest: {files[-1]}")
        
        # Check file size
        latest_size = os.path.getsize(files[-1]) / (1024*1024) # Convert to MB
        print(f"Latest checkpoint size: {latest_size:.2f} MB")
        
        # Verify file is readable
        try:
            import torch
            checkpoint = torch.load(files[-1])
            print("Latest checkpoint is loadable")
        except Exception as e:
            print(f"Warning: Could not load latest checkpoint: {e}")

if __name__ == "__main__":
    if not os.path.exists('output'):
        print("Error: 'output' directory not found!")
    else:
        check_model_checkpoints()