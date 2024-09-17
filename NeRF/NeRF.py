# import json
# import os
#
# import cv2
# import imageio.v2 as imageio
# import numpy as np
# import torch
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# args = {
#     'config': 'configs/lego.txt',
#     'expname': 'blender_paper_lego',
#     'basedir': './logs',
#     'datadir': './data/nerf_synthetic/lego',
#     'netdepth': 8,
#     'netwidth': 256,
#     'netdepth_fine': 8,
#     'netwidth_fine': 256,
#     'N_rand': 1024,
#     'lrate': 0.0005,
#     'lrate_decay': 500,
#     'chunk': 32768,
#     'netchunk': 65536,
#     'no_batching': True,
#     'no_reload': False,
#     'ft_path': None,
#     'N_samples': 64,
#     'N_importance': 128,
#     'perturb': 1.0,
#     'use_viewdirs': True,
#     'i_embed': 0,
#     'multires': 10,
#     'multires_views': 4,
#     'raw_noise_std': 0.0,
#     'render_only': False,
#     'render_test': False,
#     'render_factor': 0,
#     'precrop_iters': 500,
#     'precrop_frac': 0.5,
#     'dataset_type': 'blender',
#     'testskip': 8,
#     'shape': 'greek',
#     'white_bkgd': True,
#     'half_res': True,
#     'factor': 8,
#     'no_ndc': False,
#     'lindisp': False,
#     'spherify': False,
#     'llffhold': 8,
#     'i_print': 100,
#     'i_img': 500,
#     'i_weights': 10000,
#     'i_testset': 50000,
#     'i_video': 50000
# }
#
# trans_t = lambda t: torch.Tensor([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, t],
#     [0, 0, 0, 1]]).float()
#
# rot_phi = lambda phi: torch.Tensor([
#     [1, 0, 0, 0],
#     [0, np.cos(phi), -np.sin(phi), 0],
#     [0, np.sin(phi), np.cos(phi), 0],
#     [0, 0, 0, 1]]).float()
#
# rot_theta = lambda th: torch.Tensor([
#     [np.cos(th), 0, -np.sin(th), 0],
#     [0, 1, 0, 0],
#     [np.sin(th), 0, np.cos(th), 0],
#     [0, 0, 0, 1]]).float()
#
#
# def pose_spherical(theta, phi, radius):
#     c2w = trans_t(radius)
#     c2w = rot_phi(phi / 180. * np.pi) @ c2w
#     c2w = rot_theta(theta / 180. * np.pi) @ c2w
#     c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
#     return c2w
#
#
# def load_blender_data(basedir, half_res=False, testskip=1):
#     splits = ['train', 'val', 'test']
#     metas = {}
#     for s in splits:
#         with open(os.path.normpath(os.path.join(basedir, 'transforms_{}.json'.format(s))), 'r') as fp:
#             metas[s] = json.load(fp)
#
#     all_imgs = []
#     all_poses = []
#     counts = [0]
#     for s in splits:
#         meta = metas[s]
#         imgs = []
#         poses = []
#         if s == 'train' or testskip == 0:
#             skip = 1
#         else:
#             skip = testskip
#
#         for frame in meta['frames'][::skip]:
#             fname = os.path.normpath(os.path.join(basedir, frame['file_path'] + '.png'))
#             imgs.append(imageio.imread(fname))
#             poses.append(np.array(frame['transform_matrix']))
#         imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
#         poses = np.array(poses).astype(np.float32)
#         counts.append(counts[-1] + imgs.shape[0])
#         all_imgs.append(imgs)
#         all_poses.append(poses)
#
#     i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
#
#     imgs = np.concatenate(all_imgs, 0)
#     poses = np.concatenate(all_poses, 0)
#
#     H, W = imgs[0].shape[:2]
#     camera_angle_x = float(meta['camera_angle_x'])
#     focal = .5 * W / np.tan(.5 * camera_angle_x)
#
#     render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
#
#     if half_res:
#         H = H // 2
#         W = W // 2
#         focal = focal / 2.
#
#         imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
#         for i, img in enumerate(imgs):
#             imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
#         imgs = imgs_half_res
#         # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
#
#     return imgs, poses, render_poses, [H, W, focal], i_split
#
#
# class Embedder:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#         self.create_embedding_fn()
#
#     def create_embedding_fn(self):
#         embed_fns = []
#         d = self.kwargs['input_dims']
#         out_dim = 0
#         if self.kwargs['include_input']:
#             embed_fns.append(lambda x: x)
#             out_dim += d
#
#         max_freq = self.kwargs['max_freq_log2']
#         N_freqs = self.kwargs['num_freqs']
#
#         if self.kwargs['log_sampling']:
#             freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
#         else:
#             freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)
#
#         for _freq in freq_bands:
#             for _p_fn in self.kwargs['periodic_fns']:
#                 embed_fns.append(lambda x, p_fn=_p_fn, freq=_freq: p_fn(x * freq))
#                 out_dim += d
#
#         self.embed_fns = embed_fns
#         self.out_dim = out_dim
#
#     def embed(self, inputs):
#         return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
#
#
# # Model
# class NeRF(torch.nn.Module):
#     def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
#         """
#         """
#         super(NeRF, self).__init__()
#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.input_ch_views = input_ch_views
#         self.skips = skips
#         self.use_viewdirs = use_viewdirs
#
#         self.pts_linears = torch.nn.ModuleList(
#             [torch.nn.Linear(input_ch, W)] + [torch.nn.Linear(W, W) if i not in self.skips else torch.nn.Linear(W + input_ch, W) for i in range(D - 1)])
#
#         ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
#         self.views_linears = torch.nn.ModuleList([torch.nn.Linear(input_ch_views + W, W // 2)])
#
#         ### Implementation according to the paper
#         # self.views_linears = nn.ModuleList(
#         #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
#
#         if use_viewdirs:
#             self.feature_linear = torch.nn.Linear(W, W)
#             self.alpha_linear = torch.nn.Linear(W, 1)
#             self.rgb_linear = torch.nn.Linear(W // 2, 3)
#         else:
#             self.output_linear = torch.nn.Linear(W, output_ch)
#
#     def forward(self, x):
#         input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
#         h = input_pts
#         for i, l in enumerate(self.pts_linears):
#             h = self.pts_linears[i](h)
#             h = torch.nn.functional.relu(h)
#             if i in self.skips:
#                 h = torch.cat([input_pts, h], -1)
#
#         if self.use_viewdirs:
#             alpha = self.alpha_linear(h)
#             feature = self.feature_linear(h)
#             h = torch.cat([feature, input_views], -1)
#
#             for i, l in enumerate(self.views_linears):
#                 h = self.views_linears[i](h)
#                 h = torch.nn.functional.relu(h)
#
#             rgb = self.rgb_linear(h)
#             outputs = torch.cat([rgb, alpha], -1)
#         else:
#             outputs = self.output_linear(h)
#
#         return outputs
#
#     def load_weights_from_keras(self, weights):
#         assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
#
#         # Load pts_linears
#         for i in range(self.D):
#             idx_pts_linears = 2 * i
#             self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
#             self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))
#
#         # Load feature_linear
#         idx_feature_linear = 2 * self.D
#         self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
#         self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))
#
#         # Load views_linears
#         idx_views_linears = 2 * self.D + 2
#         self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
#         self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))
#
#         # Load rgb_linear
#         idx_rbg_linear = 2 * self.D + 4
#         self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
#         self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))
#
#         # Load alpha_linear
#         idx_alpha_linear = 2 * self.D + 6
#         self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
#         self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))
#
#
# def get_embedder(multires, i=0):
#     if i == -1:
#         return torch.nn.Identity(), 3
#
#     embed_kwargs = {
#         'include_input': True,
#         'input_dims': 3,
#         'max_freq_log2': multires - 1,
#         'num_freqs': multires,
#         'log_sampling': True,
#         'periodic_fns': [torch.sin, torch.cos],
#     }
#
#     embedder_obj = Embedder(**embed_kwargs)
#     embed = lambda x, eo=embedder_obj: eo.embed(x)
#     return embed, embedder_obj.out_dim
#
#
# def batchify(fn, chunk):
#     if chunk is None:
#         return fn
#
#     def ret(inputs):
#         return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
#
#     return ret
#
#
# def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
#     inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
#     embedded = embed_fn(inputs_flat)
#
#     if viewdirs is not None:
#         input_dirs = viewdirs[:, None].expand(inputs.shape)
#         input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
#         embedded_dirs = embeddirs_fn(input_dirs_flat)
#         embedded = torch.cat([embedded, embedded_dirs], -1)
#
#     outputs_flat = batchify(fn, netchunk)(embedded)
#     outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
#     return outputs
#
#
# def create_nerf():
#     embed_fn, input_ch = get_embedder(multires=10, i=0)
#
#     input_ch_views = 0
#     embeddirs_fn = None
#
#     use_viewdirs = True
#     if use_viewdirs:
#         embeddirs_fn, input_ch_views = get_embedder(multires=4, i=0)
#
#     N_importance = 128
#     netdepth = 8
#     netwidth = 256
#     netdepth_fine = 8
#     netwidth_fine = 256
#     output_ch = 5 if N_importance > 0 else 4
#     skips = [4]
#     model = NeRF(D=netdepth, W=netwidth,
#                  input_ch=input_ch, output_ch=output_ch, skips=skips,
#                  input_ch_views=input_ch_views, use_viewdirs=use_viewdirs).to(device)
#     grad_vars = list(model.parameters())
#
#     model_fine = None
#     if N_importance > 0:
#         model_fine = NeRF(D=netdepth_fine, W=netwidth_fine,
#                           input_ch=input_ch, output_ch=output_ch, skips=skips,
#                           input_ch_views=input_ch_views, use_viewdirs=use_viewdirs).to(device)
#         grad_vars += list(model_fine.parameters())
#
#     netchunk = 1024 * 64
#     network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
#                                                                         embed_fn=embed_fn,
#                                                                         embeddirs_fn=embeddirs_fn,
#                                                                         netchunk=netchunk)
#     lrate = 5e-4
#     optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))
#
#     start = 0
#
#     render_kwargs_train = {
#         'network_query_fn': network_query_fn,
#         'perturb': 1,
#         'N_importance': N_importance,
#         'network_fine': model_fine,
#         'N_samples': 64,
#         'network_fn': model,
#         'use_viewdirs': use_viewdirs,
#         'white_bkgd': True,
#         'raw_noise_std': 0.,
#     }
#
#     print('Not ndc!')
#     render_kwargs_train['ndc'] = False
#     render_kwargs_train['lindisp'] = False
#
#     render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
#     render_kwargs_test['perturb'] = False
#     render_kwargs_test['raw_noise_std'] = 0.
#
#     return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
#
#
# def get_rays_np(H, W, K, c2w):
#     i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
#     dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
#     # Rotate ray directions from camera frame to the world frame
#     rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
#     return rays_o, rays_d
#
#
# def train():
#     images, poses, render_poses, hwf, i_split = load_blender_data(basedir="data/nerf_synthetic/lego", half_res=True, testskip=8)
#     i_train, i_val, i_test = i_split
#
#     H, W, focal = hwf
#     H, W = int(H), int(W)
#     hwf = [H, W, focal]
#
#     K = np.array([
#         [focal, 0, 0.5 * W],
#         [0, focal, 0.5 * H],
#         [0, 0, 1]
#     ])
#
#     os.makedirs(os.path.normpath(os.path.join(basedir, expname)), exist_ok=True)
#     f = os.path.normpath(os.path.join(basedir, expname, 'args.txt'))
#     print(f)
#
#     render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf()
#     global_step = start
#
#     bds_dict = {
#         'near': 2.,
#         'far': 6.,
#     }
#     render_kwargs_train.update(bds_dict)
#     render_kwargs_test.update(bds_dict)
#
#     render_poses = torch.Tensor(render_poses).to(device)
#
#     N_rand = 1024
#     use_batching = True
#     if use_batching:
#         # For random ray batching
#         print('get rays')
#         rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
#         print('done, concats')
#         rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
#         rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
#         rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
#         rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
#         rays_rgb = rays_rgb.astype(np.float32)
#         print('shuffle rays')
#         np.random.shuffle(rays_rgb)
#
#         print('done')
#         i_batch = 0
#
#     # Move training data to GPU
#     if use_batching:
#         images = torch.Tensor(images).to(device)
#     poses = torch.Tensor(poses).to(device)
#     if use_batching:
#         rays_rgb = torch.Tensor(rays_rgb).to(device)
#
#     N_iters = 200000 + 1
#     print('Begin')
#     print('TRAIN views are', i_train)
#     print('TEST views are', i_test)
#     print('VAL views are', i_val)
#
#
# if __name__ == '__main__':
#     torch.set_default_device('cuda')
#     torch.set_default_dtype(torch.float32)
#     print(torch.get_default_device())
#     train()
