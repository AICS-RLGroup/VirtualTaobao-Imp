from utils.utils import *


def GAE(reward, value, mask, gamma, lam):
    adv = FLOAT(reward.shape[0], 1)
    delta = FLOAT(reward.shape[0], 1)

    pre_value, pre_adv = 0, 0
    for i in reversed(range(reward.shape[0])):
        # print(reward.shape[1])
        # print(mask.shape[1])
        # print(value.shape[1])
        delta[i] = reward[i] + gamma * pre_value * mask[i] - value[i]

        adv[i] = delta[i] + gamma * lam * pre_adv * mask[i]
        pre_adv = adv[i, 0]
        pre_value = value[i, 0]
    returns = value + adv
    adv = (adv - adv.mean()) / adv.std()
    return adv, returns


def PPO_step(state, action, reward, old_log_prob, policy, value, policy_optim, value_optim, epsilon):
    value_o = value(state.detach())
    done = action[:, 5:].detach()
    advantage, returns = GAE(reward.detach(), value_o.detach(), done, gamma=0.95, lam=0.95)

    value_optim.zero_grad()
    v_loss = (value_o - returns.detach()).pow(2).mean()
    for param in value.parameters():
        v_loss += param.pow(2).sum() * 1e-4
    v_loss.backward()
    value_optim.step()

    policy_optim.zero_grad()
    log_prob = policy.get_log_prob(state.detach(), action.detach())
    ratio = torch.exp(log_prob - old_log_prob.detach())
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
    p_loss = -torch.min(surr1, surr2).mean()
    p_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 40)
    policy_optim.step()
    return v_loss, p_loss
