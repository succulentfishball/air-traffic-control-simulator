import numpy as np
import random
import math


class GameState:
    def __init__(self, planes, radius=5000, destination=(0, 0, 0)):
        self.planes = planes  # List of Plane objects
        self.radius = radius  # Collision detection range
        self.destination = destination  # Destination coordinates

    # Required format:
    #  next_state, reward, done = gamestate.step(action)
    def step(self, actions):
        rewards = []
        for plane, action in zip(self.planes, actions):
            reward = plane.move(action, self.planes, self.destination, self.radius)
            rewards.append(reward)
        next_state = self.get_state()
        done = self.is_done()
        return next_state, rewards, done

    def get_state(self):
        return [(plane.x, plane.y, plane.z, plane.velocity, plane.heading) for plane in self.planes]
    # Plane is done when it reaches the destination
    def is_done(self):
        return all(plane.distance_to(self.destination) < 100 for plane in self.planes)

    def reset(self):
        for plane in self.planes:
            plane.reset()
        return self.get_state()
        
class Airplane:
    def __init__(self, id, x, y, z, velocity, heading):
        self.id = id
        self.x, self.y, self.z = x, y, z
        self.velocity = velocity
        self.heading = heading  # Angle in degrees
        self.start_pos = (x, y, z)

    def move(self, action, all_planes, destination, radius):
        # Apply action (change heading or altitude)
        if action == "left":
            self.heading -= 5
        elif action == "right":
            self.heading += 5
        elif action == "ascend":
            self.z += 100
        elif action == "descend":
            self.z -= 100

        # Convert heading to movement in x, y
        rad = math.radians(self.heading)
        self.x += self.velocity * math.cos(rad)
        self.y += self.velocity * math.sin(rad)

        # Calculate rewards
        reward = 100 - self.distance_to(destination)  # Encourage movement toward destination
        for other in all_planes:
            if other.id != self.id:
                dist = self.distance_to(other)
                if dist < radius:
                    reward -= (radius ** 2 - dist ** 2) / (radius ** 2 / 500)  # Collision penalty
        return reward

    def distance_to(self, obj):
        return math.sqrt((self.x - obj[0])**2 + (self.y - obj[1])**2 + (self.z - obj[2])**2)

    def reset(self):
        self.x, self.y, self.z = self.start_pos

# Create a SARSA agent
class SARSA_Agent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = {}
    
    def get_state(self, plane, destination, other_planes):
        # Distance from plane to other planes
        nearest_dist = min([plane.distance_to(p) for p in other_planes if p.id != plane.id], default=9999)
        # I'm not sure what the heading_diff is for
        heading_diff = (plane.heading - math.degrees(math.atan2(destination[1] - plane.y, destination[0] - plane.x))) % 360
        return (round(plane.x, -2), round(plane.y, -2), round(plane.z, -2), round(nearest_dist, -2), round(heading_diff, -1))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.actions, key=lambda a: self.q_values.get((state, a), 0))

    # Update Q-value for state-action pair using SARSA formula
    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)

    def train(self, gamestate: GameState, num_episodes=100):
        for episode in range(num_episodes):
            # Initialize state
            state = gamestate.reset()
            # Choose action - where did plane come from?
            actions = [self.choose_action(self.get_state(plane, gamestate.destination, gamestate.planes)) for plane in gamestate.planes]

            while True:
                # Take action
                next_state, reward, done = gamestate.step(action)
                # Choose next action
                next_actions = [self.choose_action(self.get_state(plane, gamestate.destination, gamestate.planes)) for plane in gamestate.planes]
                # Update Q-value
                # Q(s, a) = Q(s, a) + alpha * (reward + gamma * Q(s', a') - Q(s, a))
                # It updates q_values using previous q_values
                for i, plane in enumerate(gamestate.planes):
                    q = self.get_q_value(state[i], actions[i])
                    next_q = self.get_q_value(next_state[i], next_actions[i])
                    self.q_values[(state[i], actions[i])] = q + self.alpha * (rewards[i] + self.gamma * next_q - q)
                state, actions = next_state, next_actions
                # When all planes reach destination, break
                if done:
                    break
    print(f'For episode: {episode}, the Q table is:\n {self.q_values}')
        
if __name__ == '__main__':
    planes = [Airplane(1, 0, 0, 10000, 300, 0), Airplane(2, 100, 100, 10000, 300, 180)]
    gamestate = GameState(planes, destination=(51.47, -0.45, 0))
    agent = SARSA_Agent(actions=["left", "right", "ascend", "descend", "maintain"])
    agent.train(gamestate)


    