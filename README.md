# Samuel_Babalola_RL_Summative

Here's a comprehensive README.md for your project:

```markdown
# Indoor Navigation RL Agent - Summative Project

[![Project Video](https://drive.google.com/file/d/1spGp9PRDBwnUUeGqmQoyrbpl3z5n_hUi/view?usp=sharing)](https://www.youtube.com/watch?v=VIDEO_ID)

A reinforcement learning project comparing DQN and PPO algorithms for autonomous indoor navigation in a custom house environment.

## Table of Contents
- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Custom Environment](#custom-environment)
- [Training Process](#training-process)
- [Visualization](#visualization)
- [Results](#results)
- [Report](#report)
- [License](#license)

## Project Overview
This project implements and compares two RL approaches for indoor navigation to assist the visually imapired:
- **Value-Based Method**: Deep Q-Network (DQN)
- **Policy-Based Method**: Proximal Policy Optimization (PPO)

The agent learns to navigate through a custom house layout with multiple rooms and obstacles while maximizing rewards and minimizing collisions.

![Environment Visualization](docs/environment_preview.png)

## Installation
```bash
git clone https://github.com/SammyGbabs/Samuel_Babalola_RL_Summative.git
pip install -r requirements.txt
```

## Project Structure
```
Samuel_Babalola_RL_Summative/
├── Environment/
│   ├── custom_env.py            # Custom Gym environment
│   ├── rendering.py             # PyOpenGL visualization
├── Notebooks/                     # Saved DQN models
│   ├── Samuel_Babalola_RL_Summative_DQN.ipynb
|   └── Samuel_Babalola_RL_Summative_PPO.ipynb      
├── Training/
│   ├── dqn_training.py          # DQN training script
│   └── pg_training.py           # PPO training script
├── dqn models/
│   ├── best_models.zip          # Saved DQN models
│   └── dqn_final_model.zip      # Saved DQN models
├── ppo models/
│   ├── best_models.zip          # Saved PPO models
│   └── ppo_final_model.zip      # Saved PPO models
├── docs/                        # Report and visualizations
├── main.py                      # Main entry point
└── requirements.txt             # Dependencies
```

## Custom Environment
### Key Specifications
- **Action Space** (Discrete):
  ```python
  0: Move Up    1: Move Down
  2: Move Left   3: Move Right 
  4: Wait
  ```
- **Observation Space**:
  - 16-dimensional vector including:
  - Proximity sensors (5 values)
  - Target room encoding (4 values)
  - Navigation state (7 values)
  
- **Reward Structure**:
  - +15 for reaching target
  - +1 for doorway usage
  - -5 for collisions
  - -0.1 per step penalty

## Training Process
### DQN Training
```bash
python -m Training/dqn_training.py 
```

### PPO Training
```bash
python -m Training/pg_training.py 
```

## Visualization
Run the trained agent:
```bash
python -m main.py --model-path ./ppo_models/best_model.zip --model-type ppo --fps 2 --step-delay 0.5
```

Key visualization features:
- Real-time agent movement tracking
- Obstacle and doorway highlighting
- Path history visualization
- Multiple camera angles

## Results
### Performance Comparison
| Metric           | DQN       | PPO       |
|------------------|-----------|-----------|
| Avg. Steps       | 32.10     | 14.3      |
| Avg Rewards      | 38.86     | 41.1      |
| Training Time    | 45mins    | 1 hour    |

## Report
Key findings documented in [Final_Report.pdf](docs/Final_Report.pdf):
- PPO showed better convergence in complex layouts
- DQN performed better with sparse rewards
- Doorway bonuses improved learning speed by 40%
- Optimal γ value found to be 0.99 for both methods

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Contact**: Samuel Oluwajunwonlo Babalola | s.babalola@alustudent.com  
**Submission Date**: 1st of April 2025
```
