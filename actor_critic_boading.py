import warnings
warnings.filterwarnings("ignore")

import json
import sys
import numpy as np
import gym
import myosuite
import pandas as pd

import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

torch.manual_seed(0)
device = "mps" if torch.cuda.is_available() else "cpu"


#Building the environment
env = gym.make("myoChallengeBaodingP1-v1").unwrapped

init_state = env.reset()

# Get number of actions from gym action space
n_actions = env.action_space.shape[0]


class Critic(nn.Module):
    #TODO: Can be made broader and deeper
    def __init__(self,state_dim,act_dim):
        super(Critic,self).__init__()
        self.fc1=nn.Linear(state_dim,128)
        self.fc2=nn.Linear(128+act_dim,128)
        self.fc3=nn.Linear(128,1)
        
    def forward(self,state,action):
        x=self.fc1(state)
        x=F.relu(x)
        x=self.fc2(th.cat((x,action),1))
        x=F.relu(x)
        x=self.fc3(x)
        return x

class Actor(nn.Module):
    #TODO: Can be made broader and deeper
    def __init__(self,state_dim,act_dim,max_a):
        super(Actor,self).__init__()
        self.fc1=nn.Linear(state_dim,256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2=nn.Linear(256,128)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3=nn.Linear(128,act_dim)
        self.fc3.weight.data.normal_(0, 0.1)
        self.max_a=max_a
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return th.tanh(x)

class replay_memory():
    def __init__(self,replay_memory_size):
        self.memory_size=replay_memory_size
        self.memory=np.array([])
        self.new=0
        self.cur=0
    def size(self):
        return self.memory.shape[0]
    def store(self,trans):
        if(self.memory.shape[0]<self.memory_size):
            if self.new==0:
                self.new=1
                self.memory=np.array(trans,dtype=object)
            elif self.memory.shape[0]>0:
                self.memory=np.vstack((self.memory,trans))
        else:
            self.memory[self.cur,:]=trans
            self.cur=(self.cur+1)%self.memory_size
    
    def sample(self,batch_size):
        if self.memory.shape[0]<batch_size:
            return -1
        sam=np.random.choice(self.memory.shape[0],batch_size)
        return self.memory[sam]

class Logger:
    def __init__(self):
        self.epss = []
        self.varis = []
        self.mean_rews = []
        self.median_rews = []
        self.max_rews = []
        
    def store_log(self,values):
        self.epss.append(values[eps_index])
        self.varis.append(round(values[varis_index],3))
        self.mean_rews.append(round(np.mean(values[mean_index]),2))
        self.median_rews.append(round(np.median(values[median_index]),2))
        self.max_rews.append(round(np.max(values[median_index]),2))
    
    def save_log(self,file_name):
        self.log_df = pd.DataFrame( data = {
            'Episodes': self.epss,
            'Total Mean Rewards': self.mean_rews,
            'Total Median Rewards': self.median_rews,
            'Total Max Reward': self.max_rews,
            'Variance': self.varis
        } )
        self.log_df.to_csv(f'{file_name}.csv',index = False)


def log_decay(eps,episode,min_eps = 0.05):
    eps = eps - (1/np.exp(1/(episode+1)))
    eps = max(min_eps, eps)
    return eps

class DDPG():
    def __init__(self):
        self.actor=Actor(n_state,n_action,max_action).to(device)
        self.target_actor=Actor(n_state,n_action,max_action).to(device)
        self.critic=Critic(n_state,n_action).to(device)
        self.target_critic=Critic(n_state,n_action).to(device)
        self.memory=replay_memory(memory_size)
        self.Aoptimizer=th.optim.Adam(self.actor.parameters(),lr=lr)
        self.Coptimizer=th.optim.Adam(self.critic.parameters(),lr=lr)

    def actor_learn(self,batch):
        b_s=th.FloatTensor(np.array(batch[:,0].tolist())).to(device)
        action=self.actor(b_s)
        loss=-(self.critic(b_s,action).mean())
        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()
    
    def critic_learn(self,batch):
        b_s=th.FloatTensor(np.array(batch[:,0].tolist())).to(device)
        b_r=th.FloatTensor(np.array(batch[:,1].tolist())).to(device)
        b_a=th.FloatTensor(np.array(batch[:,2].tolist())).to(device)
        b_s_=th.FloatTensor(np.array(batch[:,3].tolist())).to(device)
        b_d=th.FloatTensor(np.array(batch[:,4].tolist())).to(device)

        next_action=self.target_actor(b_s_)
        target_q=self.target_critic(b_s_,next_action)
        for i in range(b_d.shape[0]):
            if b_d[i]:
                target_q[i]=b_r[i]
            else:
                target_q[i]=b_r[i]+gamma*target_q[i]
        eval_q=self.critic(b_s,b_a)

        td_error=eval_q-target_q.detach()
        loss=(td_error**2).mean()
        self.Coptimizer.zero_grad()
        loss.backward()
        self.Coptimizer.step()

    def soft_update(self):
        for param,target_param in zip(self.actor.parameters(),self.target_actor.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
        for param,target_param in zip(self.critic.parameters(),self.target_critic.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)



#global variables
params = {
    'lr': 0.001,
    'tau': 0.005,
    'gamma': 0.9,
    'memory_size': 5000,
    'max_t': 500,
    'batchsize': 32,
    'n_action':39,
    'n_state': env.observation_space.shape[0],
    'max_action': float(env.action_space.high[0]),
    'n_episodes': 50000,
    'warmup': 500,
}
lr,tau,gamma,memory_size,max_t,batchsize,n_action,n_state,max_action,n_episodes,warmup = params['lr'],params['tau'],params['gamma'],params['memory_size'],params['max_t'],params['batchsize'],params['n_action'],params['n_state'],params['max_action'],params['n_episodes'],params['warmup']
#Logger variables
eps_index = 0 #episode_index in logging
varis_index = 1 #variance index in logging
mean_index = 2 # mean index in logging
median_index = 2 # median index in logging
max_index = 2 #max rewards index
metric_log_interval = n_episodes/100 #keep this and the print interval as same for now
metric_store_interval = 5000
metirc_print_interval = n_episodes/100
model_save_interval = 500
experiment_name = 'boa_test5'
model_path = f'models/{experiment_name}'
metrics_path = f'metrics/{experiment_name}'
ddpg=DDPG()
logger = Logger()

def main():
    var = 3
    rewards = []
    for episode in range(n_episodes):
        s=env.reset()
        total_reward=0
        Normal=th.distributions.normal.Normal(th.FloatTensor([0]),th.FloatTensor([var]))
        t=0
        while t<max_t:
            # noise=th.clamp(Normal.sample(),env.action_space.low[0], env.action_space.high[0]).to(device)
            # noise=th.clamp(Normal.sample(),-0.3, 0.3).to(device)
            a=ddpg.actor(th.FloatTensor(s).to(device))
            a=th.clamp(a,env.action_space.low[0], env.action_space.high[0]).to(device)
            
            
            s_,r,done,_=env.step(np.array(a.tolist()))
            total_reward+=r
            transition=[s,[r],a.tolist(),s_,[done]]
            ddpg.memory.store(transition)
            if done:
                break
            s=s_
            if(ddpg.memory.size()<warmup):
                continue
            
            batch=ddpg.memory.sample(batchsize)
            ddpg.critic_learn(batch)
            ddpg.actor_learn(batch)
            ddpg.soft_update()
            t+=1
        #plot on make_dot
        # make_dot(a, params=dict(ddpg.actor.named_parameters())).render("actor", format="png")
        if episode%metirc_print_interval ==0:
            print(f"""eps:{episode}, sc:{
                        round(total_reward,2)},avg:{
                        round(np.mean(rewards),2)},var:{round(var,3)}"""
                    )
        if episode%metric_log_interval == 0:
            rewards.append(total_reward)
            logger.store_log([episode,var,rewards])
            rewards = []
        else:
            rewards.append(total_reward)
        if episode%metric_store_interval == 0:
            logger.save_log(metrics_path)
        if episode%model_save_interval ==0:
            model_scripted = torch.jit.script(ddpg.actor) # Export to TorchScript
            model_scripted.save(f'{model_path}.pt') # Save


        #add the log variance
        # var = log_decay(var,episode,min_eps=0.01)
        var = max(0.99995*var,0.005)
    logger.store_log([episode,total_reward,var,rewards])
    logger.save_log(metrics_path)

if __name__=='__main__':
    main()
    params['final_mean_reward'] = logger.mean_rews[-1]

# torch.save(ddpg.actor,model_path)
model_scripted = torch.jit.script(ddpg.actor) # Export to TorchScript
model_scripted.save(f'{model_path}.pt') # Save

#Save the hyperparameters configuration and the final reward
with open(f'{metrics_path}_params.json','w') as f:
    json.dump(params,f)