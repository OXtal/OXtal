from typing import Any, Callable, Optional

import torch

from oxtal.model.modules.diffusion import JointDiffusionModule
from oxtal.model.utils import centre_random_augmentation

_EPS = 1e-6


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps)
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Scheduler parameters

        Args:
            s_max (float, optional): maximal noise level. Defaults to 160.0.
            s_min (float, optional): minimal noise level. Defaults to 4e-4.
            rho (float, optional): the exponent numerical part. Defaults to 7.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho

    def time_to_noise_lvl(self, time: torch.Tensor) -> torch.Tensor:
        assert ((0.0 <= time) & (time <= 1.0)).all()

        return (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + time * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = self.time_to_noise_lvl(step_indices * step_size)
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    noise_scheduler: Optional[InferenceNoiseScheduler] = None,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    s_skip: Optional[torch.Tensor] = None,
    z_skip: Optional[torch.Tensor] = None,
    output_x1: bool = False,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    sigma_seq = None
    if noise_scheduler is not None:
        seq_time = torch.tensor(1, device=device)
        sigma_seq = noise_scheduler.time_to_noise_lvl(seq_time)

    def _chunk_sample_diffusion(
        chunk_n_sample, inplace_safe, x_l=None
    ):  # @NOTE: x_l input as prior if already known
        # init noise
        # [..., N_sample, N_atom, 3]
        # x_l shape:
        x_l = (
            x_l
            if not (x_l is None)
            else noise_schedule[0]
            * torch.randn(
                size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
            )
        )  # NOTE: set seed in distributed training
        # @NOTE: batch_shape needs to be empty, chunk_n_sample needs to be 1 to work

        for _, (c_tau_last, c_tau) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
            # [..., N_sample, N_atom, 3]
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )

            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            # Euler step only
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            iter_sigma_seq = None
            if sigma_seq is not None:
                iter_sigma_seq = sigma_seq * (gamma + 1)

            kwargs = {
                "x_noisy": x_noisy,
                "t_hat_noise_level_struct": t_hat,
                "t_hat_noise_level_seq": iter_sigma_seq,
                "input_feature_dict": input_feature_dict,
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "z_trunk": z_trunk,
                "s_skip": s_skip,
                "z_skip": z_skip,
                "chunk_size": attn_chunk_size,
                "inplace_safe": inplace_safe,
            }

            if isinstance(denoise_net, JointDiffusionModule):
                kwargs["compute_seq_logits"] = False

            x_denoised = denoise_net(**kwargs)
            delta = (x_noisy - x_denoised) / t_hat[
                ..., None, None
            ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

        return x_l, x_denoised

    # look at the size of chunk (what it means)
    if diffusion_chunk_size is None:
        x_l_init = input_feature_dict["x_l"] if "x_l" in input_feature_dict else None
        x_l, x_denoised = _chunk_sample_diffusion(
            N_sample, inplace_safe=inplace_safe, x_l=x_l_init
        )
    else:
        x_l = []
        x_denoised = []
        x_l_init = input_feature_dict["x_l"] if "x_l" in input_feature_dict else None
        no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size if i < no_chunks - 1 else N_sample - i * diffusion_chunk_size
            )
            chunk_x_l, chunk_x_denoised = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe, x_l=x_l_init
            )
            x_l.append(chunk_x_l)
            x_denoised.append(chunk_x_denoised)
        x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]
        x_denoised = torch.cat(x_denoised, -3)

    if output_x1:
        return x_l, x_denoised
    else:
        return x_l
