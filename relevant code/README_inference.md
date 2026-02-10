# TGCN Inference Setup

## Overview
This inference script allows you to run the trained TGCN model on pose keypoints data to predict ASL signs.

## Prerequisites
- Python environment with PyTorch installed
- Trained model checkpoint in `archived/asl100/ckpt.pth`
- Pose keypoints data in `../../data/pose_per_individual_videos/`

## Usage

### Basic Usage
```bash
python inference.py --video_id <VIDEO_ID> --frame_start <START> --frame_end <END>
```

### Example Commands
```bash
# Test with video 69241 (frames 1-64)
python inference.py --video_id 69241 --frame_start 1 --frame_end 64

# Test with video 65225 (frames 1-64)
python inference.py --video_id 65225 --frame_start 1 --frame_end 64

# Show top 10 predictions instead of 5
python inference.py --video_id 69241 --frame_start 1 --frame_end 64 --top_k 10
```

### Command Line Arguments
- `--video_id`: Video ID to run inference on (required)
- `--frame_start`: Starting frame number (default: 1)
- `--frame_end`: Ending frame number (default: 50)
- `--config`: Path to config file (default: archived/asl100/asl100.ini)
- `--checkpoint`: Path to model checkpoint (default: archived/asl100/ckpt.pth)
- `--keypoints_dir`: Directory containing pose keypoints (default: ../../data/pose_per_individual_videos)
- `--top_k`: Number of top predictions to show (default: 5)

## Available Words for Testing

The ASL100 dataset includes 100 common ASL signs. Here are some useful words for creating test sentences:

### Basic Words
- **Objects**: `book`, `computer`, `chair`, `table`, `bed`, `pizza`, `apple`
- **Actions**: `eat`, `drink`, `walk`, `play`, `study`, `work`, `cook`
- **People**: `mother`, `father`, `family`, `man`, `woman`, `doctor`
- **Animals**: `dog`, `cat`, `bird`, `fish`
- **Colors**: `blue`, `red`, `white`, `black`, `green`, `pink`
- **Questions**: `what`, `who`, `how`, `when`
- **Responses**: `yes`, `no`
- **Descriptions**: `hot`, `cold`, `big`, `small`, `tall`, `short`

### Example Sentences
- "I want to eat pizza" (want, eat, pizza)
- "My mother likes the blue book" (mother, like, blue, book)
- "The dog can walk" (dog, can, walk)
- "I need to study now" (need, study, now)
- "What time is it?" (what, time)

## Output Format
The script outputs the top-k predictions with their probabilities:
```
Top 5 predictions:
----------------------------------------
1. kiss (0.977)
2. cook (0.008)
3. color (0.004)
4. son (0.004)
5. tell (0.001)
```

## File Structure
```
WLASL/
├── code/
│   └── TGCN/
│       ├── inference.py          # This script
│       ├── test_tgcn.py          # Testing script
│       ├── train_tgcn.py         # Training script
│       ├── tgcn_model.py         # Model definition
│       ├── sign_dataset.py       # Dataset loading
│       ├── configs.py            # Configuration
│       └── archived/
│           └── asl100/
│               ├── asl100.ini    # Model config
│               └── ckpt.pth      # Trained weights
└── data/
    ├── pose_per_individual_videos/  # Pose keypoints
    └── splits/
        └── asl100.json           # Label mapping
```

## Notes
- The model expects pose keypoints in JSON format from OpenPose or similar pose estimation tools
- Keypoints should be in the format: `image_XXXXX_keypoints.json`
- The model processes 55 keypoints per frame
- Frame sampling is done linearly between start and end frames
- Missing keypoints are filled with zeros 