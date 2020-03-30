
def render_text_envq(env, agent):
    env.seed(12456)
    state = env.reset()

    while True:
        env.render()

        max_action = agent.act(state)
        state, reward, done, info = env.step(max_action)
        print(reward)
        if (done):
            print("Environment Terminated")
            break
    env.render()