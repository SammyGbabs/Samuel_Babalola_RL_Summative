import gym
import numpy as np
from gym import spaces

class IndoorNavEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.grid_size = 15  # House layout: 15x15 grid
        self.action_space = spaces.Discrete(5)  # Up, Down, Left, Right, Wait
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(12,), dtype=np.float32  # Sensor array + target info
        )
        
        # House layout configuration
        self.rooms = {
            'living_room': (0,0,6,6),
            'kitchen': (0,7,6,14),
            'bedroom': (7,0,14,6),
            'bathroom': (7,7,14,14)
        }
        self.doorways = [(6,3), (3,6), (6,10), (10,6)]
        self.obstacles = [
            (2,2), (4,5), (2,9),  # Furniture
            (8,4), (12,2), (9,12),  # Appliances
            (5,12), (13,8)  # Decorations
        ]
        self.current_room = None
        self.target_room = None
        self.agent_pos = None

    def reset(self):
        self.agent_pos = (1,1)  # Start in living room
        self.current_room = 'living_room'
        self.target_room = np.random.choice(['kitchen', 'bedroom', 'bathroom'])
        return self._get_obs()

    def _get_obs(self):
        x,y = self.agent_pos
        sensors = [
            self._is_obstacle(x+1,y),  # Front
            self._is_obstacle(x-1,y),  # Back
            self._is_obstacle(x,y-1),  # Left
            self._is_obstacle(x,y+1),   # Right
            self._is_doorway(x,y),      # Current pos
            self._room_identification() # Target room one-hot
        ]
        return np.concatenate(sensors)

    def _is_obstacle(self, x, y):
        return 1 if (x,y) in self.obstacles or not (0 <= x <15 and 0 <= y <15) else 0

    def _is_doorway(self, x, y):
        return 1 if (x,y) in self.doorways else 0

    def _room_identification(self):
        return np.array([
            int(self.target_room == 'kitchen'),
            int(self.target_room == 'bedroom'),
            int(self.target_room == 'bathroom')
        ])

    def step(self, action):
        # Movement logic with collision detection
        if action < 4:  # Movement actions
            dx, dy = [(1,0), (-1,0), (0,-1), (0,1)][action]
            new_x = np.clip(self.agent_pos[0]+dx, 0,14)
            new_y = np.clip(self.agent_pos[1]+dy, 0,14)
            
            if not self._is_obstacle(new_x, new_y):
                self.agent_pos = (new_x, new_y)

        # Reward calculation
        reward = -0.1  # Time penalty
        done = False
        
        if self.agent_pos in self.doorways:
            reward += 0.5  # Encourage door usage
        
        if self._in_target_room():
            reward = 10
            done = True
            
        if self._is_obstacle(*self.agent_pos):
            reward = -5
            done = True

        return self._get_obs(), reward, done, {}