import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import helper as h
import multiprocess as mp
from multiprocess.connection import Connection
import tqdm
import os
import random
os.environ['OPENBLAS_NUM_THREADS'] = '1'

class LearnedModel(nn.Module):
	"""Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		obs_dim=cfg.obs_shape[0]
		cfg.obs_dim=obs_dim
		self.d_ensemble = cfg.d_ensemble
		self._dynamics = h.mlp(obs_dim+cfg.action_dim, cfg.mlp_dim, obs_dim)
		self._reward = h.mlp(obs_dim+cfg.action_dim, cfg.mlp_dim, 1)
  
		if cfg.use_td:
			self._pi = h.mlp(obs_dim, cfg.mlp_dim, cfg.action_dim)
			self.Qs=nn.ModuleList([h.q(cfg),h.q(cfg)])

		self.apply(h.orthogonal_init)
		for m in [self._reward] + ([q for q in self.Qs] if cfg.use_td else []):
			m[-1].weight.data.fill_(0)
			m[-1].bias.data.fill_(0)
		self.register_buffer(
			"obs_delta_mean", torch.zeros(obs_dim, device="cuda")
		)
		self.register_buffer(
			"obs_delta_std", torch.ones(obs_dim, device="cuda")
		)

	def track_q_grad(self, enable=True):
		"""Utility function. Enables/disables gradient tracking of Q-networks."""
		for m in self.Qs:
			h.set_requires_grad(m, enable)

	def next(self, z, a):
		"""Predicts next state and single-step reward."""
		x = torch.cat([z, a], dim=-1)
		############################ Begin Code Q4.1 ############################
		# call self._dynamics and self._reward to get next state and reward
		raise NotImplementedError("Q4.1")
		############################# End Code Q4.1 #############################

	def pi(self, z, std=0):
		if self.cfg.use_td:
			"""Samples an action from the learned policy (pi)."""
			mu = torch.tanh(self._pi(z))
			if std > 0:
				std = torch.ones_like(mu) * std
				return h.TruncatedNormal(mu, std).sample(clip=0.3)
			return mu
		else:
			"""Uniformly sample an action from [-1,1]."""
			return torch.tensor(np.random.uniform(-1,1,(z.shape[0],self.cfg.action_dim))).cuda()

	def Q(self, z, a):
		"""Predict state-action value (Q)."""
		x = torch.cat([z, a], dim=-1)
		return self.Qs[0](x), self.Qs[1](x)

	def V(self,z):
		"""Predict state value (V)."""
		############################ Begin Code Q5.1 ############################
		raise NotImplementedError("Q5.1")
		############################# End Code Q5.1 #############################

	def update_statistics(self, obs, acs, next_obs):
		"""Update observation statistics used for normalization."""
		############################ Begin Code Q4.1 ############################
		# update self.obs_delta_mean and self.obs_delta_std
		raise NotImplementedError("Q4.1")
		############################# End Code Q4.1 #############################




class RealModel():
	def __init__(self,cfg,env_fn):
		self.env=env_fn()
		self.action_dim=self.env.action_space.shape[0]
		self.batch_size=int(cfg.num_samples*(1+cfg.mixture_coef))
		
		# parallelize the environment
		def _worker(remote, parent_remote, env):
			parent_remote.close()
			while True:
				try:
					s, a = remote.recv()
					ns,r=env.next(s,a)
					remote.send((ns,r))
				except EOFError:
					break
		self.workers=[]
		forkserver_available = 'forkserver' in mp.get_all_start_methods()
		start_method = 'forkserver' if forkserver_available else 'spawn'
		ctx = mp.get_context(start_method)
		self.remotes: list[Connection]; self.work_remotes: list[Connection]
  		# todo: change #process to other number to match your cpu cores
		num_workers=128
		self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(num_workers)]) 
		self.processes = []
		for work_remote, remote in tqdm.tqdm(zip(self.work_remotes, self.remotes),desc='Starting env workers',total=num_workers):
			args = (work_remote, remote, self.env)
			process = ctx.Process(target=_worker, args=args, daemon=True)
			process.start()
			self.processes.append(process)
			work_remote.close()
		
	def next(self,z,a):
		if len(z.shape)==1:
			# single input
			ns,rew=self.env.next(z.cpu().numpy(),a.cpu().numpy())
			return torch.tensor(ns).cuda(), torch.tensor(rew).cuda()
		else:
	  		# batch of inputs
			batch_size=z.shape[0]
			# launch in parallel using multiprocessing
			ns,rew=[],[]
   
			num_segments = (batch_size + len(self.work_remotes) - 1) // len(self.work_remotes)
			for i in range(num_segments):
				start = i * len(self.work_remotes)
				end = min((i + 1) * len(self.work_remotes), batch_size)
				for remote, z_i, a_i in zip(self.remotes[:end-start], z[start:end], a[start:end]):
					remote.send((z_i.cpu().numpy(), a_i.cpu().numpy()))
				for remote in self.remotes[:end-start]:
					n, r = remote.recv()
					ns.append(n)
					rew.append(r)
     
			return torch.tensor(ns).cuda(), torch.tensor(rew).cuda()[...,None]

	def pi(self,z,*args):
		# randomly sample initial actions for the MPC
		return torch.tensor(np.random.uniform(-1,1,(z.shape[0],self.action_dim))).cuda()
	   

class ModelBasedAgent():
	"""Implementation of TD-MPC learning + inference."""
	def __init__(self, cfg,env_fn):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.std = h.linear_schedule(cfg.std_schedule, 0)
		if cfg.use_real_model:
			self.model = RealModel(cfg,env_fn)
		else:
			self.model = LearnedModel(cfg).cuda()
			self.model_target = deepcopy(self.model)
			self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
			if cfg.use_td:
				self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
			self.model.eval()
			self.model_target.eval()

	def state_dict(self):
		return {'model': self.model.state_dict(),
				'model_target': self.model_target.state_dict(),
				'obs_delta_mean': self.model.obs_delta_mean,
				'obs_delta_std': self.model.obs_delta_std}

	def save(self, fp):
		"""Save state dict of TOLD model to filepath."""
		state_dict = self.state_dict()
		torch.save(state_dict, fp)
	
	def load(self, fp):
		"""Load a saved state dict from filepath into current agent."""
		d = torch.load(fp)
		self.model.load_state_dict(d['model'])
		self.model_target.load_state_dict(d['model_target'])
		self.model.obs_delta_mean = d['obs_delta_mean']
		self.model.obs_delta_std = d['obs_delta_std']

	@torch.no_grad()
	def estimate_value(self, z, actions):
		"""Estimate value of a trajectory starting at state z and executing given actions."""
		horizon=actions.shape[0]
		G=0
		############################ Begin Code Q1,Q5.1 ############################
		# for t in range(horizon):
		# 	call self.model.next to get next state and reward
		# 	accumulate reward
		# note: do not forget the discount factor
		# note for Q5: add final state value estimation
		raise NotImplementedError("Q1,Q5.1")
		############################# End Code Q1,Q5.1 #############################
		return G

	@torch.no_grad()
	def plan(self, obs, eval_mode=False, step=None, t0=True):
		"""
		Plan next action.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		"""
		# Seed steps
		if step < self.cfg.seed_steps and not eval_mode:
			return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

		# Sample policy trajectories
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step))) if not self.cfg.use_real_model else self.cfg.horizon
		if self.cfg.eval_only:
			horizon = self.cfg.horizon

		assert self.cfg.mpc_algo in ["mppi", "cem", "random"]

		num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
		if num_pi_trajs > 0:
			############################ Begin Code Q5.4 ############################
			# sample num_pi_trajs policy trajectories using self.model.pi
			# pi_actions = ???
			pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
			############################# End Code Q5.4 #############################

		if self.cfg.mpc_algo=="random":
			############################ Begin Code Q1 ############################ 
			# actions = torch.empty(???, device=self.device).uniform_(-1, 1)
			# rewards = self.estimate_value(obs, actions)
			# return ???
			raise NotImplementedError("Q1")
			############################# End Code Q1 #############################
   

		# Initialize state and parameters
		z = obs.repeat(self.cfg.num_samples+num_pi_trajs, 1)
		mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
		std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device)
		if not t0 and hasattr(self, '_prev_mean'):
			mean[:-1] = self._prev_mean[1:]

		############################ Begin Code Q2,Q3,Q5.4 ############################
		# Iterate CEM or MPPI
		# for i in range(self.cfg.iterations):
		# 	actions = ???
		# 	value = ???
		# 	elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
		# 	elite_value, elite_actions = ???
  		# 	mean, std = ???
		# return ???
  
		# hint 1: you can use torch.clamp to ensure that the actions are within the action space
		# hint 2: MPPI weights ~ torch.exp(self.cfg.temperature*(elite_value - max_elite_value))
		# hint 3: in Q5.4, add actions = torch.cat([actions, pi_actions], dim=1) to concatenate the policy actions
  
		if self.cfg.mpc_algo=="cem":
			raise NotImplementedError("CEM not implemented yet")
		if self.cfg.mpc_algo=="mppi":
			raise NotImplementedError("MPPI not implemented yet")

		############################ End Code Q2,Q3,Q5.4 ############################
	
 
	def update_pi(self, zs):
		"""Update policy using a sequence of latent states."""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		pi_loss = 0
		############################ Begin Code Q5.3 ############################
		# pi_loss = ???
		############################# End Code Q5.3 #############################

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_z, reward):
		"""Compute the TD-target from a reward and the observation at the following time step."""
		############################ Begin Code Q5.2 ############################
		# td_target = ???
		############################# End Code Q5.2 #############################
		return td_target

	def update(self, replay_buffer, step):
		"""Main update function. Corresponds to one iteration of the TOLD model learning."""
		obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.std = h.linear_schedule(self.cfg.std_schedule, step)
		self.model.train()
  
		z = obs
		zs = [obs.detach()]

		dynamics_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
		total_loss = 0
		for t in range(self.cfg.horizon):
			############################ Begin Code Q4.2, Q5 ############################
			# Compute losses
			# hint 1: losses at different timesteps are weighted by self.cfg.rho^t 
			# hint 2: recurrently predict the next state and reward from last predicted state can reduce the compounding error
			# hint 3: in Q5, when computing the TD-target, call self._td_target. Remember to detach the target.
			# hint 4: in Q5, priority_loss is the L1 loss between Q-values and TD-targets
   			# hint 5: in Q5, you should wrap your code with a check of self.cfg.use_td
			pass
			############################# End Code Q4.2, Q5 #############################
		total_loss = self.cfg.dynamics_coef * dynamics_loss.clamp(max=1e4) + \
					 self.cfg.reward_coef * reward_loss.clamp(max=1e4)
		if self.cfg.use_td:	
			total_loss+=self.cfg.value_coef * value_loss.clamp(max=1e4)
		weighted_loss = (total_loss.squeeze(1) * weights).mean()
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		weighted_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()
  
		if self.cfg.use_td:
			############################ Begin Code Q5.3 ############################
			# Update replay buffer priorities: replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())
			# Update policy by calling self.update_pi
			# Update target network by calling h.ema
			pass
			############################# End Code Q5.3 #############################

		self.model.eval()
		loss_dict={}
		############################ Begin Code ############################
		# Maintain your own log dict to keep track of losses, gradients, etc. You can then visualize these in tensorboard.
		# loss_dict = ???
		############################# End Code #############################
		return loss_dict
