import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
import streamlit as st

# This application demonstrates how different "personalities" learn to play games
# using LunarLander which is more fun to watch!

class GamePersonalityDemo:
    def __init__(self):
        # Initialize session state for action preferences
        if 'action_preferences' not in st.session_state:
            st.session_state.action_preferences = {}
    def create_env(self):
        #Create LunarLander environment
        return gym.make('LunarLander-v2')
    def get_personality_action(self, state, personality, episode_num):
        # Choose action based on AI personality
        x_position, y_position, x_velocity, y_velocity, angle, angle_velocity, left_leg, right_leg = state

        if personality == "cautious":
            # Very careful | Prefers gentle movements
            if abs(x_velocity) > 0.5 or abs(y_velocity) > 0.5:
                return 0 # Let momentum carry
            elif angle > 0.1:
                return 1 # Fire left engine
            elif angle < -0.1:
                return 3 # Fire right engine
            else:
                return 2 #Fire main engine gently
        
        elif personality == "aggressive":
            # Bold moves, uses main engine a lot
            if y_velocity < -0.5:
                return 2 # Fire main engine hard
            elif abs(angle) >0.2:
                if angle > 0: # FIre left engine
                    return 1
                else:
                    return 3 # Fire right engine
            else:
                return random.choice([0, 2])
        
        elif personality == 'learning':
            # Gets smarter over time
            state_key = (round(angle, 1), round(y_velocity,1))

            #Start random but remember what worked
            if state_key not in st.session_state.action_preferences:
                st.session_state.action_preferences[state_key] = [0,0,0,0] #for each action

            #Early episodes: more random
            # Later episodes: use learned preferences
            exploration_rate = max(0.1, 1.0 - episode_num/50)

            if random.random() < exploration_rate:
                return random.randint(0,3)
            else:
                best_action = np.argmax(st.session_state.action_preferences[state_key])
                return best_action
            
        elif personality == 'clumsy':
            # Unpredictable and wobbly
            if random.random() <0.3:
                return random.randint(0, 3) #30% completely random
            elif abs(angle) > 0.3:
                # Sometimes tries to correct, sometimes make it worse
                if random.random() < 0.6:
                    if angle > 0:
                        return 1
                    else: 
                        return 3 # Correct
                else:
                    if angle>0:
                        return 3
                    else:
                        return 1 # Wrong direction
            else:
                return 0 # Do nothing most of the time
        else: #random
            return random.randint(0, 3)
    
    def update_learning(self, state, action, reward, personality):
        #Update preferences for learning personality
        if personality == 'learning':
            angle = state[4]
            y_velocity = state[3]
            state_key = (round(angle, 1), round(y_velocity, 1))

            if state_key in st.session_state.action_preferences:
                # Reward good actions, punish bad ones
                st.session_state.action_preferences[state_key][action] += reward / 10

    def run_episode(self, env, personality, episode_num):
        # Run single episode with given personality
        state, info= env.reset()
        total_reward = 0
        steps = 0
        crashed = False
        landed = False

        for step in range(1000): #max 1000 steps
            action = self.get_personality_action(state, personality, episode_num)
            next_state, reward, terminated, truncated, info = env.step(action)
            # Update learning for smart personality
            self.update_learning(state, action, reward, personality)

            total_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                #Check if it landed successfully or crashed
                if reward > 0:
                    landed = True
                else:
                    crashed = True
                break

        return total_reward, steps, landed, crashed
    
    def run_experiment(self, personality, episodes = 30):
        # Run experiment with specified personality
        # Reset learning for fresh start
        if personality == 'learning':
            st.session_state.action_preferences = {}
        
        env = self.create_env()
        rewards = []
        steps_listed = []
        landings = 0
        crashes = 0

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        for ep in range(episodes):
            reward, steps, landed, crashed = self.run_episode(env,personality, ep)
            rewards.append(reward)
            steps_listed.append(steps)

            if landed:
                landings += 1
            if crashed:
                crashes += 1

            # Updated progress
            progress = (ep + 1) / episodes
            progress_bar.progress(progress)

            if landed:
                status = "Landed!"
            elif crashed:
                status = "Crashed!"
            else:
                status = "Timeout"

        env.close()
        progress_bar.empty()
        status_text.empty()

        return {
            'rewards': rewards,
            'steps': steps_listed,
            'episodes': episodes,
            'personality': personality,
            'landings': landings,
            'crashes': crashes
        }

    def plot_results(self, results):
        # Plot experiment results with fun visualizations
        # Fun color scheme based on personality
        colors = {
            'cautious': 'orange',
            'aggressive': 'red',
            'learning': 'green',
            'clumsy': 'purple',
            'random': 'blue'
        }

        color  = colors.get(results['personality'], 'black')

        # create 2 columns for rewards over episodes and success vs. failure
        col1, col2 = st.columns(2)

        with col1:
            # Rewards over episodes (learning cover)
            fig1, ax1 = plt.subplots(figsize = (8,6))
            episodes = range(len(results['rewards']))
            ax1.plot(episodes, results['rewards'], color=color, alpha=0.7, linewidth=2)
            ax1.set_title(f"{results['personality'].title()} Personality: Learning Progress")
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True)
            ax1.axhline(y=0, color='black', linestyle= '--', alpha = 0.5, label= 'Break Even')
            ax1.axhline(y=200, color='gold', linestyle='--', alpha = 0.7, label = 'Good Landing')
            ax1.legend()
            st.pyplot(fig1)

            # Reward distribution plot
            fig2, ax2 = plt.subplots(figsize = (8, 6))
            ax2.hist(results['rewards'], bins=15, color=color, alpha=0.7, edgecolor = 'black')
            ax2.set_title("Performance consistency")
            ax2.set_xlabel('Reward')
            ax2.set_ylabel('Frequency')
            ax2.axvline(x=np.mean(results['rewards']), color='red', linestyle='--', linewidth=2, label=f"Average: {np.mean(results['rewards']):.0f}")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

        with col2:
            # Success vs Failure
            fig3, ax3 = plt.subplots(figsize=(8,6))
            outcomes = ['Successful\nLandings', 'Crashes', 'Timeouts']
            timeouts =  results['episodes'] - results['landings'] - results['crashes']
            counts = [results['landings'], results['crashes'], timeouts]
            colors_bar = ['green', 'red', 'gray']

            bars = ax3.bar(outcomes, counts, color=colors_bar, alpha=0.7)
            ax3.set_title('Mission Outcomes')
            ax3.set_ylabel('Number of Episodes')

            #Add count labels on bars
            for bar, count in zip(bars, counts):
                if count >0:
                    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, str(count), ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig3)

            # Performance stats
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            avg_reward = np.mean(results['rewards'])
            success_rate = results['landings'] / results['episodes'] * 100
            crash_rate = results['crashes'] / results['episodes'] * 100
            avg_steps = np.mean(results['steps'])

            stats = ['Avg Score', 'Success %', 'Crash %', 'Avg Steps']
            values = [avg_reward, success_rate, crash_rate, avg_steps]

            bars = ax4.bar(stats, values, color=[color, 'green', 'red', 'blue'], alpha=0.7)
            ax4.set_title('Personality Report Card')
            ax4.set_ylabel('Value')

            for bar, value in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig4)

def main():
    st.set_page_config(page_title='Lunar Lander AI Demo', page_icon='🌙')

    st.title("Lunar Lander AI Demo 🌙")
    st.write("Watch different AI personalities try to land on the moon!")

    #Simple controls
    personality = st.selectbox("Choose AI Pilot:", ['cautious', 'aggressive', 'learning', 'clumsy', 'random'])
    episodes = st.slider("Episodes:", 10, 50, 20)

    #Run experiment
    if st.button("Launch Mission"):
        demo = GamePersonalityDemo()

        with st.spinner("Running experiment..."):
            results = demo.run_experiment(personality, episodes)

        st.success("Mission Complete!")

        #Simple results
        avg_score= np.mean(results['rewards'])
        success_rate = results['landings']/results['episodes'] * 100

        st.write(f"Average Score: {avg_score:.1f}")
        st.write(f"Success Rate: {success_rate:.1f}%")

        #Show plots
        demo.plot_results(results)
            
if __name__ == "__main__":
    main()





