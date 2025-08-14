import random

random.seed(777)
n_states = 10
actions = ['left', 'right']
Q = {(s, a): 0.0 for s in range(n_states) for a in actions}

alpha = 0.1
gamma = 0.9
eps = 0.1
n_episode = 500

def step(St, At):
    prev_dist = abs(St - 9)
    if At == 'right':
        S_next = min(St + 1, 9)
    else:
        S_next = max(St - 1, 0)

    new_dist = abs(S_next - 9)

    if S_next == 9:
        R_next = 10
        done = True
    elif new_dist < prev_dist:
        R_next = 1
        done = False
    elif new_dist > prev_dist:
        R_next = -1
        done = False
    else:
        R_next = 0
        done = False
    return S_next, R_next, done

def policy(St):
    if random.random() < eps:
        At = random.choice(actions)
    else:
        q_left = Q[(St, 'left')]
        q_right = Q[(St, 'right')]
        At = 'left' if q_left > q_right else 'right'
    return At

for _ in range(n_episode):
    St = 0
    done = False

    while not done:
        At = policy(St)
        S_next, R_next, done = step(St, At)

        maxQ_next = max(Q[(S_next, a)] for a in actions)
        Q[(St, At)] += alpha * (R_next + gamma * maxQ_next - Q[(St, At)])

        St = S_next

print('Q-learning 학습 결과 (Q-values): ')
for s in range(n_states):
    print(f'State {s}: left={Q[(s, "left")]: .2f}, right={Q[(s, "right")]: .2f}')
