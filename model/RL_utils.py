import numpy as np
import torch
import torch.nn as nn


def get_advantages(values, rewards, over_lens, positive=False):
    """

    :param values: The output of crite network
    :param masks: whether to stop
    :param rewards: the reward from env
    :return:
    """
    returns=[]
    gae=0
    gamma=0.99
    lmbda=0.95
    values=[x.detach() for x in values]
    over_lens=over_lens.reshape(1,-1)
    mvalues=[]
    for i in reversed(range(len(rewards))):
        if i==len(rewards)-1:
            mask=0
        else:
            mask=(torch.abs(rewards[i+1])>0).float()
        delta=rewards[i]+gamma*values[i+1]*mask-values[i]*(torch.abs(rewards[i])>0).float()
        mvalues.append(values[i]*(torch.abs(rewards[i])>0).float())
        gae=delta+gamma*lmbda*gae*mask
        returns.insert(0, gae+mvalues[-1])
    returns=torch.cat(returns, 0)
    adv=returns-torch.cat(mvalues, 0)
    adv=adv#*over_lens
    non_zero=over_lens.sum()+1e-5
    total=adv.shape[1]
    mean=adv.mean(1, keepdim=True)#*total*1.0/non_zero
    std=adv.std(1, keepdim=True)#(((adv-mean)**2).sum(1, keepdim=True)/non_zero)**0.5
    if positive:
        adv=((adv-mean)/(std+1e-10))
        #adv=adv.clamp(-10, 10)
        #adv=adv/abs(adv).max()
    else:
        adv=((adv-mean)/(std+1e-10)).clamp(-10,10)
    return returns, adv, over_lens


def ppo_loss(oldpolicy_probs, newpolicy_probs, advantages, rewards, values, entropy_beta=0.15):
    clipv=0.2
    critic_discount=0.5
    masks=(newpolicy_probs>0).float()
    ratio=torch.exp(torch.log(newpolicy_probs+1e-10)-torch.log(oldpolicy_probs+1e-10))
    ratio=ratio.clamp(0.01, 100.0)
    p1=ratio*advantages.detach()
    p2=ratio.clamp(1-clipv, 1+clipv)*advantages.detach()
    actor_loss=-(torch.min(p1,p2)*masks).mean()
    critic_loss=((rewards.detach()-values)**2*masks).mean()
    total_loss=critic_loss*critic_discount+actor_loss-entropy_beta*(-newpolicy_probs*torch.log(newpolicy_probs+1e-10)).mean()
    return total_loss
