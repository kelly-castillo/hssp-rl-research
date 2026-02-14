import streamlit as st
import subprocess
import sys

# Using streamlit to create an application that provides an interactive interface for tuning PPO hyperparameters
#Using CleanRL's implementation and training on CartPole-v1 environment

st.title("PPO Hyperparameter Sandbox")

st.info("This demo uses CleanRL's PPO implementation to show how learning rate, batch size, and GAE λ affect PPO’s performance on CartPole in real time.")

# Use common learning rate ranges = with multiple decimal places
learning_rate = st.slider("Learning Rate", 0.00001, 0.01, 0.001, step=0.00001, format="%.5f")
st.write(f"Current Learning Rate: **{learning_rate:.5f}**")

# Use common clip coefficients
clip_coefficient = st.slider("Clip Coefficient", 0.1, 0.5, 0.17, step=0.01, format="%.2f")
st.write(f"Current Clip Coefficient: **{clip_coefficient:.2f}**")

gae_lambda = st.slider("GAE Lambda", 0.8, 1.0, 0.95, step=0.1, format="%.2f")
st.write(f"Current GAE Lambda: **{gae_lambda:.2f}**")

# Add num_evs control (which affects batch size)
num_envs = st.selectbox("Number of Parallel Enviornments", [1, 2, 4, 8, 16], index=2)
effective_batch_size = num_envs * 128
st.write(f"Number of Environments: **{num_envs}**")
st.write(f"Effective Batch Size: **{effective_batch_size}** (num_envs * 128)")

# Make a total timesteps option with various options 
total_timesteps = st.selectbox("Total Timesteps", [25000, 50000, 100000, 200000], index=2) #make sure it is at 100k
st.write(f"Training for: **{total_timesteps:,}** timesteps")


if st.button("Train Agent"):
    st.write("Training started... ")
    
    # Construct the CleanRL PPO training command
    # Uses subprocess to run the training script with specified parameters
    training_command = [sys.executable, "-m", "cleanrl.ppo", "--env-id", "CartPole-v1", "--learning-rate", str(learning_rate), "--clip-coef", str(clip_coefficient), "--gae-lambda",str(gae_lambda),"--total-timesteps", str(total_timesteps), "--num-envs", str(num_envs),"--track", "False"]
    # syntax: python executable path, run PPO module, specify environment, set learning rate, clip coefficient, training duration, disable experiment tracking

    # Execute the training command and capture output
    result = subprocess.run(training_command, capture_output=True, text=True)
    
    # If training is completed successfully
    if result.returncode == 0:
        st.write("Training completed successfully!")

        # Parse training output to extract episode scores
        training_output = result.stdout
        episode_scores = []
        all_output_lines = training_output.split('\n')

        # Extract episodic returns from training logs
        for line in all_output_lines:
            if 'episodic_return=' in line:

                # Parse the episodic return value from log line
                parts = line.split('episodic_return=')
                if len(parts) > 1:
                    score_text = parts[1].split(',')[0].split()[0]
                    try:
                        score = float(score_text)
                        episode_scores.append(score)
                    except ValueError:
                        st.write("Not able to read score from: ", score_text)
        
        # Display results if episodes were found
        episodes_num = len(episode_scores)
        if episodes_num > 0:
            st.write(f"Trained for {len(episode_scores)} episodes")

            # Create line chart showing scores over time then calculate key performance metrics
            st.line_chart(episode_scores)

            recent_scores = episode_scores[-10:]
            final_average = sum(recent_scores) / len(recent_scores)
            max_score = max(episode_scores)
            min_recent = min(recent_scores)
            max_recent = max(recent_scores)
            
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Final Average Score", f"{final_average:.1f}")
            with col2:
                st.metric("Max Score", f"{max_score:.1f}")
            with col3:
                st.metric("Recent Score Range", f"{min_recent:.0f}-{max_recent:.0f}")

            # Analyze performance and provide hyperparameter tuning suggestions

            # Check if performance is high and stable
            if final_average > 450:
                if max_recent - min_recent < 50:
                    st.write("Problem solved! Great hyperparameters!")
                else:
                    st.warning("High average but still unstable. Try lowering learning rate.")

            # Unstable performance case
            elif max_score > 450:
                st.warning("Agent reached high scores but couldn't maintain them. Try a lower learning rate or a different clip coefficient.")
            
            # Has not reached target level
            else:
                st.write("Not solved yet. Try adjusting the sliders.")

            # Additional specific recommendations based on perfomance level
            if final_average < 100:
                st.write("Try increasing learning rate or training steps")
            elif final_average < 450 and final_average > 200 and max_score > 400:
                st.write("Performance is unstable. Try lower learning rate(0.0001-0.0002)")
            
            #Batch size specific advice
            st.write("---")
            st.write("**Hyperparameter Effects: **")
            if effective_batch_size >= 1024:
                st.write("Large batch size | More stable gradient but slower learning")
            elif effective_batch_size <= 256:
                st.write("Small batch size | Faster Learning but noiser gradients")
            else:
                st.write("Medium batch size | Balanced trade-off")
            
            # Gae Lambda advice
            if gae_lambda >= 0.98:
                st.write("High GAE Lambda | Considers longer-term rewards, may be slower to learn")
            elif gae_lambda <= 0.90:
                st.write("Low GAE Lambda | Focuses on immediate rewards, faster but potentially less stable")
            else:
                st.write("Balanced GAE Lambda | Good Trade-off between bias and variance")

        else:
            st.write("No episode scores found")
    else: #Display error info
        st.error("Training failed")
        st.write("Error details: ")
        st.write(result.stderr)