print("****************************")
print("Loading Dueling Q Learning Util")
print("****************************")
def render_text_envq(env, agent, brain_name):
    #env.seed(12456)
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    
    
    
    
    
    
    '''
    while True:
        #env.render()

        max_action = agent.act(state)
        state, reward, done, info = env.step(max_action)
        print(reward)
        if (done):
            print("Environment Terminated")
            break
    env.render()
    '''
    

    while True:
        action = agent.act(state)
        env_info = env.step(action.astype(int))[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score+=reward
        state = next_state
        if done: break

    print('Score:', score)