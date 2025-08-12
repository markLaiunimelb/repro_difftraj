import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from utils.Traj_UNet import *
from utils.config import args
from utils.utils import *

temp = {}
for k, v in args.items():
    temp[k] = SimpleNamespace(**v)

config = SimpleNamespace(**temp)

unet = Guide_UNet(config).cuda()
# load the model
# prePath = 'D:/MyProjects/PythonAbout/DiffusionModel/DiffTraj'
unet.load_state_dict(torch.load('./DiffTraj/Porto_steps=500_len=200_0.05_bs=512/models/07-28-18-11-15/unet_200.pt'))
# %%
n_steps = config.diffusion.num_diffusion_timesteps
beta = torch.linspace(config.diffusion.beta_start,
                      config.diffusion.beta_end, n_steps).cuda()
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)
lr = 2e-4  # Explore this - might want it lower when training on the full dataset

eta = 0.0
timesteps = 100
skip = n_steps // timesteps
seq = range(0, n_steps, skip)

# load head information for guide trajectory generation
batchsize = 500
head = np.load('./porto_heads.npy',
               allow_pickle=True)
head = torch.from_numpy(head).float()
dataloader = DataLoader(head, batch_size=batchsize, shuffle=True, num_workers=0)

# Porto
# the mean and std of head information, using for rescaling
# departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed
hmean=[0, 150.478916, 0.036027, 732.050255, 49.80335, 0.000777, 0.000777]
hstd=[1,75.333881, 0.042320, 683.865596, 45.59104, 0.001228 ,0.001228]
mean = np.array([-8.616159095166733, 41.1593785482151])
std = np.array([0.05720024008103975, 0.07409172834956572])
# the original mean and std of trajectory length, using for rescaling the trajectory length
len_mean = 732.050255 # porto
len_std = 683.865596  # porto

Gen_traj = []
Gen_head = []
for i in tqdm(range(1)):
    head = next(iter(dataloader))
    lengths = head[:, 3]
    lengths = lengths * len_std + len_mean
    lengths = lengths.int()
    tes = head[:,:6].numpy()
    Gen_head.extend((tes*hstd+hmean))
    head = head.cuda()
    # Start with random noise
    x = torch.randn(batchsize, 2, config.data.traj_length).cuda()
    ims = []
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        with torch.no_grad():
            pred_noise = unet(x, t, head)
            # print(pred_noise.shape)
            x = p_xt(x, pred_noise, t, next_t, beta, eta)
            if i % 10 == 0:
                ims.append(x.cpu().squeeze(0))
    trajs = ims[-1].cpu().numpy()
    trajs = trajs[:,:2,:]
    # resample the trajectory length
    for j in range(batchsize):
        new_traj = resample_trajectory(trajs[j].T, lengths[j])
        new_traj = new_traj * std + mean
        Gen_traj.append(new_traj)
    break

plt.figure(figsize=(8, 8))
for i in range(len(Gen_traj)):
    traj = Gen_traj[i]
    plt.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.1)
plt.tight_layout()
plt.title('gen_wkhj_traj')
plt.grid(True)
plt.savefig('gen_wkhj_traj.png')
plt.show()
