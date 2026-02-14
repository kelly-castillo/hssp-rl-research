import gymnasium as gym
import safety_gymnasium
import matplotlib.pyplot as plt
import numpy as np

# This application demonstrates the trade-off between reward maximization and safety constraints
# in RL using SafetyPointGoal1-v0 environment

# Try to import Jupyter widgets for interactive dashboard
# Go back to command-line mode if widgets aren't available
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    JUPYTER_MODE = True

except ImportError:
    JUPYTER_MODE = False

class SafetyConceptDemo:
    def __init__(self):

        # Create safety gym environment
        self.env = gym.make('SafetyPointGoal1-v0', disable_env_checker=True)
        
        # Unwrap environment if it's wrapped
        if hasattr(self.env, 'env'):
            self.env = self.env.env
    
    def run_safety_episode(self, cost_limit, safety_factor):

        # Reset environment to initial state
        obs, info = self.env.reset()
        total_reward = 0
        total_cost = 0
        
        # Run episode for a max of 1000 steps
        for step in range(1000):
            action = self.env.action_space.sample() # sample random action from action space
            
            # Apply safety factor by scaling down actions
            # Higher safety factor = more conservative = smaller actions
            action_scale = 1.0 - (safety_factor * 0.8)
            action = action * action_scale

            # execute action in environment
            step_result = self.env.step(action)
            obs, reward, cost, terminated, truncated, info = step_result

            # collect rewards and costs
            total_reward += reward
            total_cost += cost
            
            if terminated or truncated:
                break #end episode if conditions are reached

        return total_reward, total_cost
    
    def run_experiment(self, cost_limit, safety_factor, episodes=50):

        # Initialize storage for experiment data
        rewards = []
        costs = []
        violations = 0

        # Display experiment arrangement
        print(f"Running {episodes} episodes...")
        print(f"Cost limit:  {cost_limit}")
        print(f"Safety factor: {safety_factor:.2f} (0=aggressive, 1=conservative)")
        print(f"Action scaling: {1.0 - (safety_factor * 0.8):.2f}")
        print("-" * 40)

        # Run the specified number of episodes
        for ep in range(episodes):
            reward, cost = self.run_safety_episode(cost_limit, safety_factor) # run single ep
            rewards.append(reward)
            costs.append(cost)

            if cost > cost_limit: # count constraint violations
                violations += 1
            
            if (ep + 1) % 10 == 0: # print every 10 episodes
                print(f"Episode {ep + 1}: reward={reward:.1f}, cost={cost:.1f}")
        
        # return results dictionary
        return {'rewards': rewards, 'costs': costs, 'violations': violations, 'episodes': episodes, 'cost_limit': cost_limit, 'safety_factor': safety_factor}
    
    def plot_results(self, results):

        # Create 2 subplot layout for comprehensive analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(12,8))

        episodes = range(len(results['rewards']))

        # Plot 1: Reward Over Episodes
        ax1.plot(episodes, results['rewards'], 'b-', alpha=0.7)
        ax1.set_title('Reward Over Episodes')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)

        # Plot 2: Cost Over Episodes with Safety Limit
        ax2.plot(episodes, results['costs'], 'r-', alpha=0.7)
        ax2.axhline(y=results['cost_limit'], color='red', linestyle='--', label='Cost Limit') # adding horizontal line showing the cost limit constraint
        ax2.set_title('Cost Over Episodes')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Reward vs  Cost Trade-off Scatter Plot
        ax3.scatter(results['costs'], results['rewards'], alpha=0.6, c=list(episodes), cmap='viridis') # Color points by episode number to show progression
        ax3.axvline(x=results['cost_limit'], color='red', linestyle='--', label='Cost Limit') # Cost limit as vertical line
        ax3.set_xlabel('Cost')
        ax3.set_ylabel('Reward')
        ax3.set_title('Reward vs Cost Trade-off')
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Safety Performance Bar Chart
        # Calculate safe episodes (those within cost limit)
        safe = results['episodes'] - results['violations']
        ax4.bar(['Safe', 'Violations'], [safe, results['violations']], color=['green', 'red'], alpha=0.7)
        ax4.set_title('Safety Performance')
        ax4.set_ylabel('Episodes')
        ax4.grid(True)

        # Add title with important parameters
        plt.suptitle(f"Safety Demo: Cost Limit={results['cost_limit']}, Safety Factor={results['safety_factor']:.2f}")
        plt.tight_layout()
        plt.show()

        # Summary stats
        # Calculate and display performance metrics
        violation_rate = results['violations']/results['episodes'] * 100
        print(f"\nResults: ")
        print(f"Average Reward: {np.mean(results['rewards']):.2f}")
        print(f"Average Cost: {np.mean(results['costs']):.2f}")
        print(f"Violation Rate: {violation_rate:.1f}%")

    def create_dashboard(self):

        # Check if Jupyter widgets are available
        if not JUPYTER_MODE:
            print("Dashboard requires Jupyter")
            return
        
        # Display dashboard header and instructors
        print("Safety Concept Demo")
        print("Environment: SafetyPointGoal1-v0")
        print("Goal: Understand reward vs safety trade-offs")
        print("Higher safety factor = More conservative, lower rewards")
        print("Lower safety factor = More aggressive, higher rewards")
        print("-" * 50)
 
        # Interactive controls
        cost_limit_slider = widgets.FloatSlider(value=25.0, min=10.0, max=50.0, step=1.0, description='Cost Limit:')
        safety_slider = widgets.FloatSlider(value=0.2, min=0.0, max=1.0, step=0.1, description='Safety Factor:', tooltip='0=Aggressive, 1=Very Conservative')
        episodes_slider = widgets.IntSlider(value=30, min=10, max=100, step=10, description='Episodes: ')

        run_button = widgets.Button(description="Run Experiment") # execution button
        output = widgets.Output() # output area for results and plots

        def start_training(button):

            with output:
                clear_output(wait=True)
                print("Starting experiments...")
                try:
                    # Run experiment with current parameter values
                    results = self.run_experiment(cost_limit_slider.value, safety_slider.value, episodes_slider.value)
                    # Display results
                    self.plot_results(results)
                except Exception as e:
                    print(f"Error during experiment: {e}")
        # Connect button to callback function
        run_button.on_click(start_training)

        return widgets.VBox([cost_limit_slider, safety_slider, episodes_slider, run_button, output])

# Main execution
if __name__=="__main__":
    demo = SafetyConceptDemo()
    # Choose execution mode based on environment
    if JUPYTER_MODE:
        # Interactive Jupyter dashboard mode
        dashboard = demo.create_dashboard()
        display(dashboard)

    else:
        print("Safety Concept Demo")
        results = demo.run_experiment(cost_limit=25.0, safety_factor=0.3, episodes=20) # Run experiment with default parameters
        demo.plot_results(results)





