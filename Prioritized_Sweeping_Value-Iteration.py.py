def prioritized_sweeping_value_iteration1(env, optimal_values, preprocess_dict, iterations=5e5, epsilon=1e-3, gamma=0.9):
    iterations_values, crt_values = gauss_seidel_value_iteration(env, optimal_values, iterations=1, epsilon=epsilon, gamma=gamma)

    q = PriorityQueue()
    p = PriorityQueue()

    iter = 0
    break_flag = False
    while iter < iterations - 1:
        if break_flag:
            break

        max_threshold = -np.inf

        previous_values = deepcopy(crt_values)
        
        if p.empty():
            for i in range(env.observation_space.n):
                q.put((-preprocess_dict[i][1], i))

            while not q.empty():
                _, crt_state = q.get()

                max_state_val = -np.inf
                for action in range(env.action_space.n):
                    crt_reward = None
                    crt_sum = 0

                    for prob, next_state, reward, _ in env.P[crt_state][action]:
                        if crt_reward is None:
                            crt_reward = reward
                        
                        crt_sum += prob * previous_values[next_state]

                    max_state_val = max(max_state_val, crt_reward + gamma * crt_sum)

                crt_values[crt_state] = max_state_val
                iterations_values.append(compare_values(crt_values, optimal_values))
                crt_threshold = abs(crt_values[crt_state] - previous_values[crt_state])

                max_threshold = max(max_threshold, crt_threshold)

                p.put((-crt_threshold, crt_state))
            
            iter += 1

        previous_values = deepcopy(crt_values)

        while not p.empty():
            crt_threshold, crt_state = p.get()

            crt_threshold = abs(crt_threshold)
            if crt_threshold < epsilon:
                break_flag = True
                break

            max_state_val = -np.inf
            for action in range(env.action_space.n):
                crt_reward = None
                crt_sum = 0

                for prob, next_state, reward, _ in env.P[crt_state][action]:
                    if crt_reward is None:
                        crt_reward = reward
                        
                    crt_sum += prob * previous_values[next_state]

            max_state_val = max(max_state_val, crt_reward + gamma * crt_sum)

            crt_values[crt_state] = max_state_val
            iterations_values.append(compare_values(crt_values, optimal_values))
            previous_values = deepcopy(crt_values)

            crt_threshold = abs(crt_values[crt_state] - previous_values[crt_state])

            max_threshold = max(max_threshold, crt_threshold)
            p.put((-crt_threshold, crt_state))
            
            iter += 1

        if max_threshold < epsilon:
            break        

    return iterations_values, crt_values

def prioritized_sweeping_value_iteration2(env, optimal_values, preprocess_dict, iterations=5e5, epsilon=1e-3, gamma=0.9):
    iterations_values, crt_values = gauss_seidel_value_iteration(env, optimal_values, iterations=1, epsilon=epsilon, gamma=gamma)

    p = PriorityQueue()

    for state in range(env.observation_space.n):
        p.put((-crt_values[state], state))

    for k in range(int(iterations) - 1):
        previous_values = deepcopy(crt_values)

        if p.empty():
            break
        
        # print(k)

        while not p.empty():
            crt_threshold, crt_state = p.get()
            previous_value = previous_values[crt_state]

            if abs(crt_threshold) < epsilon/100:
                break

            max_value = -np.inf

            for action in range(env.action_space.n):
                crt_reward = None
                crt_sum = 0

                for prob, next_state, reward, _ in env.P[crt_state][action]:
                    if crt_reward is None:
                        crt_reward = reward
                    
                    crt_sum += prob * previous_values[next_state]

                max_value = max(max_value, crt_reward + gamma * crt_sum)

            crt_values[crt_state] = max_value
            iterations_values.append(compare_values(crt_values, optimal_values))
            
            previous_values = deepcopy(crt_values)
            p.put((-abs(crt_values[crt_state] - previous_value), crt_state))

    return iterations_values, crt_values

def prioritized_sweeping_value_iteration3(env, optimal_values, preprocess_dict, iterations=5e5, epsilon=1e-3, gamma=0.9):
    initialise_value = 1
    iterations_values, crt_values = gauss_seidel_value_iteration(env, optimal_values, iterations=initialise_value, epsilon=epsilon, gamma=gamma)

    p = PriorityQueue()

    for state in range(env.observation_space.n):
        p.put((-crt_values[state], state))
    
    previous_values = deepcopy(crt_values)

    while not p.empty() and k < int(iterations) - initialise_value:
        print(k)

        if max_error(p.queue) < epsilon:
            break

        crt_threshold, crt_state = p.get()
        previous_value = previous_values[crt_state]

        max_value = -np.inf

        for action in range(env.action_space.n):
            crt_reward = None
            crt_sum = 0

            for prob, next_state, reward, _ in env.P[crt_state][action]:
                if crt_reward is None:
                    crt_reward = reward
                
                crt_sum += prob * previous_values[next_state]

            max_value = max(max_value, crt_reward + gamma * crt_sum)

        crt_values[crt_state] = max_value
        iterations_values.append(compare_values(crt_values, optimal_values))

        state_improvement = abs(crt_values[crt_state] - previous_value)
        if state_improvement > 0:
            p.put((-state_improvement, crt_state))
        
        k += 1

        if preprocess_dict[crt_state][1] > 0:
            for predecessor in preprocess_dict[crt_state][0]:
                pred_max_value = -np.inf
                for a_prime in range(env.action_space.n):
                    pred_reward = None
                    pred_sum = 0

                    for prob, next_state, reward, _ in env.P[crt_state][a_prime]:
                        if pred_reward is None:
                            pred_reward = reward
                        
                        pred_sum += prob * crt_values[next_state]

                    pred_max_value = max(pred_max_value, pred_reward + gamma * pred_sum)

                crt_values[predecessor] = pred_max_value
                iterations_values.append(compare_values(crt_values, optimal_values))

                k += 1

                pred_state_improvement = abs(crt_values[predecessor] - previous_values[predecessor])
                if pred_state_improvement > 0:
                    p.put((-pred_state_improvement, predecessor))

        previous_values = deepcopy(crt_values)

    return iterations_values, crt_values
