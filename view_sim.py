"""
view_sim.py — Watch the combat sim with a trained model or random agent.

USAGE
    python view_sim.py --stage 3 --render human
    python view_sim.py --stage 3 --render video
    python view_sim.py --stage 3 --model checkpoints/ppo_best.pt --render video
    python view_sim.py --stage 3 --model models/v1/Combat_Large.onnx --render video
    python view_sim.py --stage 3 --arena_size 4000 --render video

REQUIRES
    pip install pygame
    For video export: pip install imageio[ffmpeg]
"""

import argparse
import sys
import os

import numpy as np
import torch

from combat_sim import CombatEnv, OBS_SIZE
from combat_extensions import make_extended_curriculum_env as make_curriculum_env
from frame_stack import FrameStackEnvWrapper, stacked_obs_size

sys.path.insert(0, os.path.dirname(__file__))


def load_onnx(onnx_path: str):
    """Load an ONNX model via onnxruntime.

    Returns (session, frame_stack).  Frame stack is inferred from
    the input tensor shape — e.g. 594 floats → 594 / 198 = 3 frames.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError(
            "onnxruntime not installed — pip install onnxruntime")

    session = ort.InferenceSession(onnx_path)
    input_shape = session.get_inputs()[0].shape
    # shape is typically [batch, obs_dim].  obs_dim may be symbolic.
    obs_dim = input_shape[1] if isinstance(input_shape[1], int) else None
    frame_stack = max(1, obs_dim // OBS_SIZE) if obs_dim else 3

    print(f"ONNX model loaded: {onnx_path}")
    print(f"  Input shape: {input_shape}, frame_stack={frame_stack}")
    return session, frame_stack


def _sample_np(logits):
    """Softmax + sample in numpy (for ONNX inference path)."""
    logits = logits - logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    return int(np.random.choice(len(probs), p=probs))


def load_model(model_path: str, device: torch.device):
    """Load a BC, PPO, or policy-only checkpoint — auto-detects architecture.

    Handles three architectures:
        1. Flat MLP (current) — actor_backbone.0.weight etc, no encoder.*
        2. Structured encoder (previous) — encoder.* keys present
        3. Old BC flat — shared.0.weight etc
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    frame_stack = ckpt.get("frame_stack", 3)
    tier = ckpt.get("tier", "large")
    input_size = stacked_obs_size(frame_stack)

    state_dict = ckpt.get("full_state_dict",
                 ckpt.get("model_state_dict",
                 ckpt.get("policy_state_dict", {})))

    has_value = any("value_head" in k for k in state_dict)
    has_encoder = any("encoder." in k for k in state_dict)
    arch_tag = ckpt.get("architecture", "structured" if has_encoder else "flat_mlp")

    print(f"Loading model: tier={tier}, frame_stack={frame_stack}, "
          f"arch={arch_tag}, "
          f"{'actor-critic' if has_value else 'policy-only'}")

    if has_encoder:
        # ── Old structured encoder architecture ──
        if has_value:
            # Full ActorCritic with structured encoder — load from old PPO module.
            # This path exists for backwards compat with old checkpoints.
            try:
                from importlib.machinery import SourceFileLoader
                ppo_mod = SourceFileLoader("ppo", "03_ppo_train.py").load_module()
                model = ppo_mod.ActorCritic(obs_size=input_size, hidden=256,
                                            frame_stack=frame_stack)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"  Warning: could not load structured ActorCritic: {e}")
                print(f"  Falling back to policy-only load...")
                from combat_policy import make_policy
                model = make_policy(tier, frame_stack=frame_stack)
                # Partial load — skip encoder/critic keys that don't match.
                own = model.state_dict()
                for k, v in state_dict.items():
                    dst = k.replace("actor_backbone", "backbone").replace("movement_head", "move_head")
                    if dst in own and v.shape == own[dst].shape:
                        own[dst] = v
                model.load_state_dict(own, strict=False)
        else:
            from combat_policy import make_policy
            model = make_policy(tier, frame_stack=frame_stack)
            model.load_state_dict(state_dict, strict=False)
    else:
        # ── Flat MLP architecture (current) ──
        if has_value:
            # Full ActorCritic (PPO training checkpoint).
            from importlib.machinery import SourceFileLoader
            ppo_mod = SourceFileLoader("ppo", "03_ppo_train.py").load_module()
            model = ppo_mod.ActorCritic(obs_size=input_size, hidden=128,
                                        tier=tier)
            model.load_state_dict(state_dict)
        else:
            # Policy-only (CombatPolicy or old BC).
            has_shared = any("shared." in k for k in state_dict)
            if has_shared:
                # Old BC architecture.
                from importlib.machinery import SourceFileLoader
                bc_mod = SourceFileLoader("bc", "01_behavioural_cloning.py").load_module()
                model = bc_mod.make_model(tier, obs_size=input_size)
                model.load_state_dict(state_dict)
            else:
                from combat_policy import make_policy
                model = make_policy(tier, frame_stack=frame_stack)
                model.load_state_dict(state_dict)

    model.eval().to(device)

    # [Fix] Load observation normalizer if present in checkpoint.
    obs_normalizer = None
    if "obs_normalizer" in ckpt:
        from importlib.machinery import SourceFileLoader
        ppo_mod = SourceFileLoader("ppo", "03_ppo_train.py").load_module()
        obs_normalizer = ppo_mod.RunningNormalizer(input_size)
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        print(f"  Loaded observation normalizer (count={obs_normalizer.count:.0f})")

    return model, frame_stack, obs_normalizer


def main():
    parser = argparse.ArgumentParser(description="Watch the combat sim")
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--archetype", type=str, default="ranged")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .pt checkpoint or .onnx model (omit for random actions)")
    parser.add_argument("--render", type=str, default="video",
                        choices=["human", "video", "none"],
                        help="'human' = pygame window, 'video' = save mp4/gif, 'none' = stats only")
    parser.add_argument("--arena_size", type=float, default=None)
    parser.add_argument("--weapon", type=str, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=str, default="sim_replay.mp4",
                        help="Output video filename (for --render video)")
    args = parser.parse_args()

    device = torch.device("cpu")

    model = None
    onnx_session = None
    frame_stack = 1
    obs_normalizer = None  # [Fix] Observation normalizer from training

    if args.model:
        if args.model.endswith(".onnx"):
            onnx_session, frame_stack = load_onnx(args.model)
            # Try loading normalizer from a companion .pt checkpoint
            # named the same as the .onnx but with .pt extension.
            pt_path = args.model.replace(".onnx", ".pt")
            if os.path.exists(pt_path):
                ckpt = torch.load(pt_path, map_location=device, weights_only=False)
                if "obs_normalizer" in ckpt:
                    from importlib.machinery import SourceFileLoader
                    ppo_mod = SourceFileLoader("ppo", "03_ppo_train.py").load_module()
                    input_size = stacked_obs_size(frame_stack)
                    obs_normalizer = ppo_mod.RunningNormalizer(input_size)
                    obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
                    print(f"  Loaded normalizer from {pt_path}")
        else:
            model, frame_stack, obs_normalizer = load_model(args.model, device)
        print(f"Model loaded. Deterministic={args.deterministic}"
              f"{', normalizer=YES' if obs_normalizer else ', normalizer=MISSING (raw obs)'}")
    else:
        print("No model — using random actions.")

    if args.render == "human":
        env_render_mode = "human"
    elif args.render == "video":
        env_render_mode = "rgb_array"
    else:
        env_render_mode = None

    if args.render == "video":
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    raw_env = make_curriculum_env(args.stage, args.archetype,
                                  render_mode=env_render_mode)
    if args.arena_size is not None:
        raw_env.cfg.arena_size = args.arena_size
    if args.weapon is not None:
        raw_env.cfg.weapon_preset = args.weapon

    if model or onnx_session:
        env = FrameStackEnvWrapper(raw_env, frame_stack=frame_stack)
    else:
        env = raw_env

    raw_env._render_size = 700

    # ── Run episodes ──
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            if model or onnx_session:
                # [Fix] Normalize observations — model was trained on normalised inputs.
                obs_input = obs
                if obs_normalizer:
                    obs_input = obs_normalizer.normalize(obs)

                # [Fix] Build action masks — prevents invalid actions
                # (fire without ammo, reload when full, target empty slots).
                mask_dict = raw_env.build_action_mask()
                m_mask_np = mask_dict["m_mask"]
                c_mask_np = mask_dict["c_mask"]
                t_mask_np = mask_dict["t_mask"]

                if onnx_session:
                    # ── ONNX inference ──
                    onnx_obs = obs_input.astype(np.float32).reshape(1, -1)
                    ort_out = onnx_session.run(None, {"observation": onnx_obs})
                    m_logits_np = ort_out[0][0]
                    c_logits_np = ort_out[1][0]
                    t_logits_np = ort_out[2][0]

                    # Apply masks.
                    m_logits_np[~m_mask_np] = -1e8
                    c_logits_np[~c_mask_np] = -1e8
                    t_logits_np[~t_mask_np] = -1e8

                    if args.deterministic:
                        m = int(np.argmax(m_logits_np))
                        c = int(np.argmax(c_logits_np))
                        t = int(np.argmax(t_logits_np))
                    else:
                        m = _sample_np(m_logits_np)
                        c = _sample_np(c_logits_np)
                        t = _sample_np(t_logits_np)
                else:
                    # ── PyTorch inference ──
                    obs_t = torch.from_numpy(obs_input).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(obs_t)
                        if len(outputs) == 4:
                            m_logits, c_logits, t_logits, _ = outputs
                        else:
                            m_logits, c_logits, t_logits = outputs

                    # Apply masks.
                    m_mask_t = torch.from_numpy(m_mask_np).unsqueeze(0)
                    c_mask_t = torch.from_numpy(c_mask_np).unsqueeze(0)
                    t_mask_t = torch.from_numpy(t_mask_np).unsqueeze(0)
                    m_logits = m_logits.masked_fill(~m_mask_t, -1e8)
                    c_logits = c_logits.masked_fill(~c_mask_t, -1e8)
                    t_logits = t_logits.masked_fill(~t_mask_t, -1e8)

                    if args.deterministic:
                        m = m_logits.argmax(dim=-1).item()
                        c = c_logits.argmax(dim=-1).item()
                        t = t_logits.argmax(dim=-1).item()
                    else:
                        m = torch.distributions.Categorical(logits=m_logits).sample().item()
                        c = torch.distributions.Categorical(logits=c_logits).sample().item()
                        t = torch.distributions.Categorical(logits=t_logits).sample().item()

                action = np.array([m, c, t])
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            if args.render == "video":
                frame = raw_env.render()
                if frame is not None:
                    frames.append(frame)

        print(f"Episode {ep+1}: reward={total_reward:.1f}, steps={steps}, "
              f"info={info}")

    # ── Save video ──
    if args.render == "video" and frames:
        try:
            import imageio
            print(f"Saving {len(frames)} frames to {args.output}...")
            if args.output.endswith(".gif"):
                imageio.mimsave(args.output, frames, fps=args.fps, loop=0)
            else:
                imageio.mimsave(args.output, frames, fps=args.fps)
            print(f"Saved: {args.output}")
        except ImportError:
            print("imageio not installed — cannot save video. "
                  "pip install imageio[ffmpeg]")


if __name__ == "__main__":
    main()