import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import random
import os

# This application demonstrates emergent behvior in multi-agent systems where predators (red)
# hunt prey (blue) in an environment with simple movement rules and collision detection

# Streamlit setup
st.title("Multi-Agent Predator-Prey Demo")
st.write("Watch predators hunt prey using teamwork")

st.sidebar.header("Team Setup")
predators = st.sidebar.slider("Predators", 1, 6, 3) # Number of red predator agents
prey = st.sidebar.slider("Prey", 2, 10, 5) # Number of blue prey agents
episodes = st.sidebar.slider("Episodes", 5, 20, 10) # How many simulations to run
st.sidebar.header("Agent Settings")
# Teamwork factor affects coordination
teamwork = st.sidebar.slider("Predator Teamwork", 0.0, 1.0, 0.5)

# Agent class definition
class Agent:
    def __init__(self, horizontal, vertical, team):
        self.horizontal = horizontal # X position
        self.vertical = vertical # Y position 
        self.team = team # Team color/type
        self.alive = True # Life status

# Game class definition   
class Game:
    def __init__(self):

        self.agents = [] # List to store all agents
        self.size = 10 # Environment size (10x10 grid)

    def add_agents(self, num_red, num_blue):
        self.agents = [] # Clear any existing agents
 
        # Create predator agents (red team)
        for i in range(num_red):
            # Random position within environment bounds
            horizontal = random.uniform(0, self.size)
            vertical = random.uniform(0, self.size)
            agent = Agent(horizontal, vertical, "red")
            self.agents.append(agent)

        # Create prey agents (blue team)
        for i in range(num_blue):
            # Random position within environment bounds
            horizontal = random.uniform(0, self.size)
            vertical = random.uniform(0, self.size)
            agent = Agent(horizontal, vertical, "blue")
            self.agents.append(agent)
    
    def distance(self, agent1, agent2):
        # Calculate Euclidean distance between two agents
        distance_horizontal = agent1.horizontal - agent2.horizontal
        distance_vertical = agent1.vertical - agent2.vertical
        return (distance_horizontal*distance_horizontal + distance_vertical*distance_vertical) ** 0.5
    
    def find_closest(self, agent, target_team):
        # Find the closest aliv eagent from specific team
        closest = None
        min_distance = 999 # Large initial value

        # Search through all agents
        for other in self.agents:
            if other.team == target_team and other.alive:
                dist = self.distance(agent, other)
                if dist < min_distance:
                    min_distance = dist
                    closest = other

        return closest
    
    def move_red(self, agent, teamwork_lvl):
        # More predator toward closest prey
        target = self.find_closest(agent, "blue") # Find closest prey to chase

        if target:
            # Calculate direction vector toward target
            distance_horizontal = target.horizontal - agent.horizontal
            distance_vertical = target.vertical - agent.vertical
            distance = (distance_horizontal*distance_horizontal + distance_vertical*distance_vertical) ** 0.5

            # Move toward target if not already at same position
            if distance > 0:
                speed = 0.3 # Predator movement speed
                # Normalize direction and apply speed
                agent.horizontal = agent.horizontal + (distance_horizontal / distance) * speed
                agent.vertical = agent.vertical + (distance_vertical/ distance) * speed
                
        # Enforce environment boundaries
        if agent.horizontal < 0:
            agent.horizontal = 0
        if agent.horizontal > self.size:
            agent.horizontal = self.size
        if agent.vertical < 0:
            agent.vertical = 0
        if agent.vertical > self.size:
            agent.vertical = self.size

    def move_blue(self, agent):
        # Move prey away from closest predator
        threat = self.find_closest(agent, "red") # Find closest predator to avoid

        if threat: # Calculate direction vector away from threat
            distance_horizontal = agent.horizontal - threat.horizontal
            distance_vertical = agent.vertical - threat.vertical
            distance = (distance_horizontal*distance_horizontal + distance_vertical*distance_vertical) ** 0.5

            # Move away from threat if not at same position
            if distance > 0:
                speed = 0.25 # Prey movement speed (slower than predators)
                # Normalize direction and apply speed
                agent.horizontal = agent.horizontal + (distance_horizontal / distance) * speed
                agent.vertical = agent.vertical + (distance_vertical/ distance) * speed
        # Enforce environment boundaries        
        if agent.horizontal < 0:
            agent.horizontal = 0
        if agent.horizontal > self.size:
            agent.horizontal = self.size
        if agent.vertical < 0:
            agent.vertical = 0
        if agent.vertical > self.size:
            agent.vertical = self.size

    def checking_catches(self):
        # Check for collisions between predators and prey
        catches = 0
        
        # Check all predator-prey paisr
        for red in self.agents:
            if red.team == "red":
                for blue in self.agents:
                    if blue.team == "blue" and blue.alive:
                        # Catch occurs if distance is less than catch radius
                        if self.distance(red, blue) < 0.5:
                            blue.alive = False
                            catches += 1

        return catches
    
    
    def count_alive(self, team):
        #Count how many agents of a specific team are still alive
        count = 0

        for agent in self.agents:
            if agent.team == team and agent.alive:
                count += 1

        return count
   
    def run_game(self):
        # Run a complete game simulation without animation
        
        total_catches = 0
        
        # Run simulation for max of 50 time steps
        for step in range(50):
            # Move all alive agents according to their behavior
            for agent in self.agents:
                if agent.alive:
                    if agent.team == "red":
                        self.move_red(agent, teamwork)
                    else:
                        self.move_blue(agent)
            # Check for new catches this step
            catches = self.checking_catches()
            total_catches = total_catches + catches

            # End game early if all prey are caught
            if self.count_alive("blue") == 0:
                break

        # Return final game statistics
        survivors = self.count_alive("blue")
        return total_catches, survivors
    
    def run_animated_game(self, steps=50):

        # Run game simulation while recording positions for animation
        # Should be same as run_game() but stores position history for each step
        
        self.position_history = []
        total_catches = 0

        # Run simulaiton and record positions
        for step in range(steps):
            # Record current positoins of all agents
            current_positions = {
                'red_horizontal': [agent.horizontal for agent in self.agents if agent.team == 'red'],
                'red_vertical' : [agent.vertical for agent in self.agents if agent.team == 'red'],
                'blue_horizontal': [agent.horizontal for agent in self.agents if agent.team == 'blue' and agent.alive],
                'blue_vertical' : [agent.vertical for agent in self.agents if agent.team == 'blue' and agent.alive]
            }
            self.position_history.append(current_positions)

            # Move all alive agents
            for agent in self.agents:
                if agent.alive:
                    if agent.team == 'red':
                        self.move_red(agent, teamwork)
                    else:
                        self.move_blue(agent)

            # Check for catches
            catches = self.checking_catches()
            total_catches = total_catches + catches

            if self.count_alive('blue') == 0: # end early if prey is caught
                break

        survivors = self.count_alive('blue')
        return total_catches, survivors
    

    def create_animation(self, filename='predator_prey.gif'):
        # Create animated GIF from recorded position history

        #Check if position data is available
        if not hasattr(self, 'position_history'):
            st.error('No animation data available. Run animated game first!')
            return False
        
        # Animation setup

        # Create figure and axis for animation
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_title("Multi-Agent Predator-Prey Animation")
        ax.set_xlabel("X Position")
        ax.set_ylabel('Y position')

        # Create scatter plots for predator and prey
        predator_scatter = ax.scatter([], [], c='red', s=100, label='Predators', marker = 'o')
        prey_scatter = ax.scatter([], [], c='blue',s=80, label='Prey', marker = 's')
        ax.legend()

        def animate_frame(frame_num):
            # Animation function called for each frame
            # Updates the positons of predators and prey based on the recorded position history

            if frame_num < len(self.position_history):
                positions = self.position_history[frame_num]

                # Update predator positons
                if positions['red_horizontal'] and positions['red_vertical']:
                    predator_data = list(zip(positions['red_horizontal'],positions['red_vertical']))
                    predator_scatter.set_offsets(predator_data)

                # Update prey positons (alive)
                if positions['blue_horizontal'] and positions['blue_vertical']:
                    prey_data = list(zip(positions['blue_horizontal'], positions['blue_vertical']))
                    prey_scatter.set_offsets(prey_data)

                else:
                    # No alive prey (clear the scatterplot)
                    prey_scatter.set_offsets([])

                return predator_scatter, prey_scatter
            
        # Create animation object
        num_frames = len(self.position_history)
        animate = animation.FuncAnimation(fig, animate_frame, frames=num_frames, interval=200, blit=True, repeat = True)

        # Save animation as GIF
        try:
            animate.save(filename, writer=PillowWriter(fps=5))
            plt.close(fig) # clean up to prevent memory leaks
            return True
        except Exception as e:
            st.error(f"Error saving animation: {e}")
            plt.close(fig)
            return False
        
# Create two-column layout for different simulation modes        
col1, col2 = st.columns(2)

# Left column: batch simulation
with col1:
    if st.button("Start Simulation"):
        st.write("Running games...")

        all_catches = []
        all_survivors = []

        # Create game instance
        game = Game()

        # Run multiple episodes to analyze
        for episode in range(episodes):
            # Reset agents for each episode
            game.add_agents(predators, prey)
            # Run single episode
            catches, survivors = game.run_game()
            # Store results
            all_catches.append(catches)
            all_survivors.append(survivors)

        # Reuslts display and analysis
        st.subheader("results")

        # Display average performance metrics
        col1, col2 = st.columns(2)

        with col1:
            avg_catches = sum(all_catches) / len(all_catches)
            st.metric("Average Catches", f'{avg_catches:.1f}')

        with col2:
            avg_survivors = sum(all_survivors) / len(all_survivors)
            st.metric('Average Survivors', f'{avg_survivors:.1f}')

        # Plot catches per episode    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        episode_numbers = list(range(1, episodes + 1))
        ax1.plot(episode_numbers, all_catches, 'ro-')
        ax1.set_title("Catches per Episode")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Catches")
        ax1.grid(True)

        # Plot survivors per episode
        ax2.plot(episode_numbers, all_survivors, 'bo-')
        ax2.set_title("Survivors per Episode")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Survivors")
        ax2.grid(True)

        plt.tight_layout()
        st.pyplot(fig)

        # Game analysis and interpretation

        st.subheader("What happened?")

        # Calculate predator success rate
        success_rate = (avg_catches / prey) * 100
        
        # Provide outocme interpretation
        if success_rate > 70:
            st.write("Predators won! They caught most prey.")
        elif success_rate > 30:
            st.info("Balanced game. Both teams did well.")
        else:
            st.warning("Prey escaped! They were too fast.")

        # Determine team balance effects
        if predators > prey:
            st.write("Many predators vs few prey = Easy hunting")
        elif predators * 2 < prey:
            st.write("Few predators vs many prey = Hard hunting")
        else:
            st.write("Balanced teams = Fair game")

# Right Column: animated simulation
with col2:
    if st.button("Create Animated GIF"):
        st.write("Creating animation...")

        # Debug info for file systems
        st.write(f"Current directory: {os.getcwd()}")
        st.write(f"Files in directory: {os.listdir('.')}")

        # Create and run animated game
        game = Game()
        game.add_agents(predators, prey)

        # Run game with position recording (shorter for animation)
        catches, survivors = game.run_animated_game(steps=30)

        # Create animation file
        success = game.create_animation("C:/Users/kelly/Desktop/predator_prey_demo.gif")

        # Display animation results
        if success:
            st.write("Animation created successfully.")
            st.write("GIF saved as 'predator_prey_demo.gif'")
            st.write(f"This episode: {catches} catches, {survivors} survivors")

            # Show episode-specific metrics
            st.subheader("Animation Episode Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Catches Made", catches)
            with col2:
                st.metric("Prey Survived", survivors)
        else:
            st.error("Failed to create animation")
    





                






    








