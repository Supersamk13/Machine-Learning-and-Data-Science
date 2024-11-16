import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Layer, BatchNormalization
import cv2
import matplotlib.pyplot as plt
from threading import Thread

# Define a custom layer to implement Dueling Architecture in Q-Learning
class DuelingOutputLayer(Layer):
    """Custom Keras layer to compute the dueling output for Q-values."""

    def __init__(self, **kwargs):
        super(DuelingOutputLayer, self).__init__(**kwargs)

    def call(self, inputs):
        V = inputs[0]  # Value stream for Q-value calculation
        A = inputs[1]  # Advantage stream for Q-value calculation
        # Calculate Q-values by combining Value and Advantage streams
        Q = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))
        return Q

# Define the main agent with Dueling Double DQN and Prioritized Experience Replay
class DuelingDoubleDQLAgentPER:
    def __init__(self, state_size, action_size, alpha_initial=0.6, alpha_final=1.0, total_episodes=1000):
        # Environment parameters
        self.state_size = state_size  # Dimension of the state space
        self.action_size = action_size  # Dimension of the action space

        # Agent parameters
        self.memory = []  # Experience memory with priority weights
        self.memory_capacity = 500000  # Max size of memory
        self.min_memory = 2500  # Minimum experiences needed for training
        self.gamma = 0.99  # Discount factor for future rewards
        self.learning_rate = 0.0001  # Learning rate for the model
        self.dropout_rate = 0.2  # Dropout rate for preventing overfitting

        # Prioritized experience replay parameters
        self.alpha_initial = alpha_initial  # Initial alpha value for prioritization
        self.alpha_final = alpha_final  # Final alpha value for prioritization
        self.total_episodes = total_episodes  # Total number of episodes for training
        self.epsilon = 1e-4  # Small constant to ensure no experience has zero probability

        # Initialize model architecture
        self.model = self._build_model()  # Main DQN model
        self.target_model = self._build_model()  # Target DQN model
        self.update_target_model()  # Sync target model with main model

    def _get_current_alpha(self, episode):
        """Calculate the linearly increasing alpha for current episode."""
        return self.alpha_initial + (self.alpha_final - self.alpha_initial) * (episode / self.total_episodes)

    def _build_model(self):
        """Build and compile the Dueling Double DQN model."""
        state_input = Input(shape=(self.state_size,))  # State input layer

        # Hidden layers with ELU activation for more stable training
        x = Dense(512, activation='elu')(state_input)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(256, activation='elu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = BatchNormalization()(x)

        x = Dense(128, activation='elu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        # Separate streams for Value and Advantage
        value = Dense(1, activation='linear')(x)  # Value stream
        advantage = Dense(self.action_size, activation='linear')(x)  # Advantage stream

        # Combine streams with custom layer to compute Q-values
        model = tf.keras.Model(inputs=state_input, outputs=DuelingOutputLayer()([value, advantage]))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipvalue=1.0))
        return model

    def update_target_model(self):
        """Update target model by copying weights from the main model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience with priority score in memory."""
        # Compute the initial priority for this new experience
        max_priority = max([exp[0] for exp in self.memory], default=1.0) if self.memory else 1.0
        # Add experience with the highest current priority to ensure it will be sampled
        self.memory.append((max_priority, (state, action, reward, next_state, done)))

        # Keep memory size within the defined capacity
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)

    def get_priority(self, error, episode):
        """Calculate priority based on the TD-error and the linearly increasing alpha."""
        alpha = self._get_current_alpha(episode)
        return (error + self.epsilon) ** alpha

    def sample_memory(self, batch_size):
        """Sample experiences based on priority."""
        # Calculate the sum of priorities for sampling probabilities
        priorities = np.array([exp[0] for exp in self.memory])
        sampling_probs = priorities / priorities.sum()

        # Sample batch indices according to probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=sampling_probs)
        samples = [self.memory[i][1] for i in indices]

        # Return both the samples and their indices for updating priorities later
        return samples, indices

    def update_priorities(self, indices, errors):
        """Update priorities in memory after each replay based on new TD-errors."""
        for idx, error in zip(indices, errors):
            self.memory[idx] = (self.get_priority(error), self.memory[idx][1])

    def softmax(self, x):
        """Calculate softmax probabilities for actions based on Q-values."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def act(self, state):
        """Choose an action using softmax on Q-values."""
        act_values = self.model.predict(state)  # Get Q-values from model
        probabilities = self.softmax(act_values[0])  # Calculate action probabilities
        return np.random.choice(self.action_size, p=probabilities)  # Select action

    def replay(self, batch_size):
        """Sample experiences from memory, update model, and adjust priorities."""
        if len(self.memory) < self.min_memory:
            return

        minibatch, indices = self.sample_memory(batch_size)

        states = np.vstack([s[0] for s in minibatch])
        next_states = np.vstack([s[3] for s in minibatch])

        # Predict Q-values for current states and next states
        target_q = self.model.predict(states)
        target_q_next = self.model.predict(next_states)
        target_q_target = self.target_model.predict(next_states)

        errors = []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            # Calculate the updated Q-value
            target = reward if done else reward + self.gamma * target_q_target[i][np.argmax(target_q_next[i])]
            errors.append(abs(target - target_q[i][action]))
            target_q[i][action] = target  # Update Q-value for chosen action

        # Update model with adjusted Q-values
        self.model.fit(states, target_q, epochs=1, verbose=0)

        # Update experience priorities based on the TD-error
        self.update_priorities(indices, errors)

def plot_moving_average(rewards, window_size=25):
    """Plot the moving average of rewards to visualize performance."""
    smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(range(window_size - 1, len(rewards)), smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Total Reward')
    plt.title('Smoothed Total Reward over Episodes')
    plt.grid()
    plt.show()

def train_agent(agent, episodes=1000):
    """Train the agent using the CartPole environment."""
    state_size = env.observation_space.shape[0]
    batch_size = 256
    rewards = []

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False

        while not done:
            # Render the environment and display the frame every 20 episodes
            if e % 20 == 0:
                frame = env.render()  # Get the current frame of the environment
                if frame is not None:  # Check if the frame is valid
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert color from RGB to BGR
                    position = state[0][0]  # Get the position from the state
                    angle = state[0][2]  # Get the angle from the state
                    time_step = total_reward  # Use total reward as time step for display
                    # Create a text overlay with the agent's position and angle
                    text = f"Position: {position:.2f}, Angle: {angle:.2f}, Time: {time_step:.2f}"
                    cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Add text to frame
                    cv2.imshow('CartPole', frame_bgr)  # Display the frame
                if cv2.waitKey(100) & 0xFF == ord('q'):  # Check for 'q' key press to quit
                    break

            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}")
        rewards.append(total_reward)

        if len(agent.memory) >= agent.min_memory:
            agent.replay(batch_size)

        if e % 50 == 0:
            agent.update_target_model()

        if (e + 1) % 25 == 0:
            Thread(target=plot_moving_average, args=(rewards,)).start()

    # Close the rendering window
    cv2.destroyAllWindows()

# Initialize environment and agent
env = gym.make('CartPole-v1', render_mode='rgb_array')
agent = DuelingDoubleDQLAgentPER(env.observation_space.shape[0], env.action_space.n)

# Start training
train_agent(agent, episodes=1000)
