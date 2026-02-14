import streamlit as st
import matplotlib.pyplot as plt
import random
import numpy as np

# This application demonstrates how curriculum learning can dramatically improve AI agent training efficiency compared to fixed difficulty training
# Uses a simulated CartPole environment to show learning curves

st.title("Curriculum Learning Playground")
st.write("See how automated difficulty scaling improves learning in CartPole!")

#Curriculum strategy selection
st.sidebar.header("Curriculum Settings")
curriculum_type = st.sidebar.selectbox("Learning Approach", ["Fixed Difficulty", "Easy Curriculum", "Gradual Curriculum", "Adaptive Curriculum"])

# Curriculum parameters
starting_difficulty = st.sidebar.slider("Starting difficulty", 0.1, 1.0, 0.3, 0.1)
ep_per_lvl = st.sidebar.slider("Episode per Level", 10, 100, 50, 10)
total_episodes = st.sidebar.slider("Total Episodes", 100, 1000, 500, 50)

# Visualization options
st.sidebar.header("CartPole Parameters")
show_pole_length = st.sidebar.checkbox("Show Pole Length Changes", True)
show_episode_steps = st.sidebar.checkbox("Show Episode Length Changes", False)

# Cartpole curriculum simulator class
class CartPoleCurriculum:
    # Simulates curriculum learning in a CartPole-like environment
 
    def __init__(self):
        # Initialize the curriculum simulator with base parameters
        self.base_pole_length = 0.5 # Base pole length in meters
        self.base_max_steps = 200 # Base max episode length
        self.reset()

    def reset(self):
        #Reset simulate state for new experiment
        self.current_difficulty = 0.5 # Current difficulty level (0-1)
        self.episode_scores= [] # Performance history
        self.difficulty_history = [] # Difficulty progression
        self.pole_lengths = [] # Pole length over time
        self.max_steps_history = [] # Max steps allowed over time

    def get_pole_length(self, difficulty):
        #Calculate pole length based on difficulty level
        # Scale from 0.5x to 2.0x base length based on difficulty
        return self.base_pole_length * (0.5 + difficulty * 1.5)
    
    def get_max_steps(self, difficulty):
        # Calculate maximum episode steps based on difficulty level
        # Scale from 1.5x to 1.0x base steps based on difficulty
        return int(self.base_max_steps * (1.5 - difficulty * 0.5))
    
    def simulate_episode(self, difficulty, episode_num, total_episodes):
        # Simulate a single CartPole episode at given

        # Get environment parameters for this difficulty
        pole_length = self.get_pole_length(difficulty) 
        max_steps = self.get_max_steps(difficulty)

        learning_bonus = episode_num / total_episodes * 0.3
        base_performance = max(0.1, 1.0 - difficulty * 0.7 + learning_bonus)

        # Add random variation to simulate learning noise
        noise = random.uniform(-0.2, 0.2)
        performance = max(0.05, min(1.0, base_performance + noise))

        # Convert performance percentage to actual episode score
        score = int(performance * max_steps)

        return score, pole_length, max_steps

    def run_curriculum(self, curriculum_type, starting_diff, ep_per_lvl, tot_ep):
        # Run a complete curriculum learning experiment
        
        # Reset simulator state
        self.reset()

        # Initialize curriculum variables
        episode = 0
        current_difficulty = starting_diff
        episodes_at_lvl = 0

        # Run episodes according to curriculum strategy
        while episode < total_episodes:
            if curriculum_type == "Fixed Difficulty":
                # Use high difficulty (hard from the start)
                current_difficulty = 0.8
            elif curriculum_type == "Easy Curriculum":
                # Easy then hard
                if episode < total_episodes // 2:
                    current_difficulty = 0.3 # Easy first half
                else:
                    current_difficulty = 0.8 # Hard second half
            elif curriculum_type == "Gradual Curriculum":
                # Linear difficulty progress from start to end
                progress = episode / total_episodes
                current_difficulty = starting_diff + (0.8 - starting_diff) * progress

            elif curriculum_type == "Adaptive Curriculum":
                # Dynamic difficulty adjustment based on recent performance 
                if len(self.episode_scores) >= 10:
                    # Calculate recent average performance
                    recent_avg = sum(self.episode_scores[-10:]) / 10
                    # Expected score for current difficulty (70% of max possible)
                    expected_score = self.get_max_steps(current_difficulty) * 0.7
                    # Increase difficulty if performing well
                    if recent_avg > expected_score and current_difficulty < 0.9:
                        current_difficulty = min(0.9, current_difficulty + 0.05)
                    elif recent_avg < expected_score * 0.5 and current_difficulty > 0.2:
                        current_difficulty = max(0.2, current_difficulty - 0.02)

            #Simulate single episode at current difficult
            score, pole_lengths, max_steps = self.simulate_episode(current_difficulty, episode, total_episodes)

            # Store all metrics for analysis
            self.episode_scores.append(score)
            self.difficulty_history.append(current_difficulty)
            self.pole_lengths.append(pole_lengths)
            self.max_steps_history.append(max_steps)

            episode += 1

# Main experiment execution
simulator = CartPoleCurriculum()

# Primary experiment button
if st.button("Run Curriculum Learning Experiment"):
    st.write("Running Simulation...")

    # Execute the curriculum learning experiment
    simulator.run_curriculum(curriculum_type, starting_difficulty, ep_per_lvl, total_episodes)

    # Prepare data for visualization
    episodes = list(range(1, len(simulator.episode_scores)+ 1))
    # Display key performance metrics
    st.subheader("Learning Results")

    # Three-column layout for key metrics: final score, average of last 50 eps, final difficulty level
    col1, col2, col3 = st.columns(3)

    with col1:
        final_score = simulator.episode_scores[-1]
        st.metric("Final Episode Score", f"{final_score}")

    with col2:
        avg_last_50 = sum(simulator.episode_scores[-50:]) / min(50, len(simulator.episode_scores))
        st.metric("Average Last 50 Episodes", f"{avg_last_50}")
    
    with col3:
        final_difficulty = simulator.difficulty_history[-1]
        st.metric("Final Difficulty Level", f"{final_difficulty:.2f}")


    # Create subplot layout based on selected visualization options
    if show_pole_length and show_episode_steps:
        # Show all 4 plots in 2x2 grid
        fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (14, 10))
    elif show_pole_length or show_episode_steps:
        # Show 3 plots in a single row
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    else:
        # Show only core 2 plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,5))

    # Plot raw episode scores
    ax1.plot(episodes, simulator.episode_scores, 'b-', alpha=0.7, linewidth=1)

    # Add moving average trend line for better look
    window = 20
    if len(simulator.episode_scores) >= window:
        moving_avg = []
        for i in range(window -1 , len(simulator.episode_scores)):
            avg = sum(simulator.episode_scores[i - window + 1: i + 1]) / window
            moving_avg.append(avg)

        # Plot moving average as red line
        ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'{window}-episode average')
        ax1.legend()

    ax1.set_title("Learning Progress")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Score")
    ax1.grid(True)

    # Plot difficulty progression over time
    ax2.plot(episodes, simulator.difficulty_history, 'g-', linewidth=2)
    ax2.set_title("Difficulty Progression")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Difficulty Level")
    ax2.set_ylim(0, 1)  # Difficulty is always 0-1
    ax2.grid(True)

    # Pole length visualization
    if show_pole_length:
        # Determine which axis to use based on layout
        ax_idx = ax3 if show_pole_length and show_episode_steps else ax3 if show_pole_length or show_episode_steps else None
        if ax_idx is not None:
            ax_idx.plot(episodes, simulator.pole_lengths, 'orange', linewidth=1.5)
            ax_idx.set_title("Pole Length Over Time")
            ax_idx.set_xlabel("Episode")
            ax_idx.set_ylabel("Pole Length (meters)")
            ax_idx.grid(True)
    
    # Episode step limits
    if show_episode_steps:
        # Determine which axis to use based on layout
        ax_idx = ax4 if show_pole_length and show_episode_steps else ax3 if not show_pole_length else None
        if ax_idx is not None:
            ax_idx.plot(episodes, simulator.max_steps_history, 'purple', linewidth=1.5)
            ax_idx.set_title("Max Episode Steps Over Time")
            ax_idx.set_xlabel("Episode")
            ax_idx.set_ylabel("Max Steps Allowed")
            ax_idx.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

    # Curriculum strategy analysis

    st.subheader("Curriculum Analysis")

    # Calculate learning improvement metric
    first_100 = simulator.episode_scores[:100]
    last_100 = simulator.episode_scores[-100:]
    improvement = (sum(last_100) / len(last_100)) - (sum(first_100) / len(first_100))

    # Two-column layout for analysis
    col1, col2 = st.columns(2)

    with col1:
        st.write("Learning Efficiency:")
        if curriculum_type == "Fixed Difficulty":
            st.write("Hard from start | May struggle to learn")
        elif curriculum_type == "Easy Curriculum":
            st.write("Two-phase learning | Sudden difficulty jump")
        elif curriculum_type == "Gradual Curriculum":
            st.write("Smooth progression | Steady improvement")
        elif curriculum_type == "Adaptive Curriculum":
            st.write("Smart adaptation | Adjusts to agent's ability")
    
    with col2:
        st.write("Performance Metrics:")
        st.write(f"Improvement: +{improvement:.1f} points")

        #Interpret improvement level
        if improvement > 50:
            st.write("Excellent learning progress!")
        elif improvement > 20:
            st.write("Good learning progress!")
        else:
            st.write("Limited learning progress")

    #Comparative analysis
    st.subheader("Compare Different Curriculums")
    if st.button("Run All Curriculum Types"):
        st.write("Comparing all curriculum approaches...")

        # List of all curriculum strategies to compare
        curriculums = ["Fixed Difficulty", "Easy Curriculum", "Gradual Curriculum", "Adaptive Curriculum"]
        all_results = {}

        # Run each curriculum type with same parameters
        for curr_type in curriculums:
            sim = CartPoleCurriculum()
            sim.run_curriculum(curr_type, starting_difficulty, ep_per_lvl, total_episodes)
            all_results[curr_type] = sim.episode_scores

        # Create comparison plot with moving averages
        fig, ax = plt.subplots(figsize = (12, 6))

        # color scheme for different curriculums
        colors = ['red', 'orange', 'green', 'blue']

        # Plot moving average for each curriculum type
        for i, (curr_type, scores) in enumerate(all_results.items()):
            episodes = list(range(1, len(scores) + 1))

            # Calculate moving average for smoother comparison
            window = 20
            if len(scores) >= window:
                moving_avg = []
                for j in range(window-1, len(scores)):
                    avg = sum(scores[j-window+1:j+1]) / window
                    moving_avg.append(avg)
                
                # Plot curriculum with distinct color
                ax.plot(episodes[window-1:], moving_avg, color=colors[i], linewidth=2, label=curr_type)
        
        ax.set_title("Curriculum Learning Comparison (20-Episode Moving Average)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Score")
        ax.legend()
        ax.grid(True)
    
        plt.tight_layout()
        st.pyplot(fig)

        # performance summary table
        st.subheader("Performance Summary")

        # Calculate summary statistics for each curriculum
        summary_data = []
        for curr_type, scores in all_results.items():
            final_avg = sum(scores[-50:]) / 50 # Average of last 50 episodes
            total_scores = sum(scores) # Cumulative performance
            summary_data.append([curr_type, f"{final_avg:.1f}", f"{total_scores:,.0f}"])

        # Display results in a clean table
        import pandas as pd
        df = pd.DataFrame(summary_data, columns = ["Curriculum Type", "Final 50 Avg", "Total Score"])
        st.table(df)




