"""
Create frame-stacked replay buffer for IQL training.

This script converts the original 72D replay buffer to a frame-stacked version.
Frame stacking provides temporal context to help the policy distinguish causality.

Original state: [front_enc(64) + circle_det(2) + joints(6)] = 72D
Stacked state: 72D x N_frames = e.g., 288D for 4 frames

Usage:
    python create_frame_stacked_replay_buffer.py --stack_size 4
    python create_frame_stacked_replay_buffer.py --stack_size 3 --input replay_buffer_iql_72d.npz
"""

import argparse
import numpy as np
from tqdm import tqdm


def create_frame_stacked_buffer(
    input_path: str,
    output_path: str,
    stack_size: int = 4,
):
    """
    Create frame-stacked replay buffer.

    Args:
        input_path: Path to original 72D replay buffer
        output_path: Path to save stacked replay buffer
        stack_size: Number of frames to stack (e.g., 4 means 72D x 4 = 288D)
    """
    print(f"Loading replay buffer from {input_path}...")
    data = np.load(input_path)

    # Handle different key names
    if "obs" in data:
        obs = data["obs"]
    else:
        obs = data["observations"]

    actions = data["actions"]
    rewards = data["rewards"]

    if "next_obs" in data:
        next_obs = data["next_obs"]
    else:
        next_obs = data["next_observations"]

    if "dones" in data:
        dones = data["dones"]
    else:
        dones = data["terminals"]

    # Load episode info if available
    episode_index = data.get("episode_index", None)

    N, obs_dim = obs.shape
    act_dim = actions.shape[1]

    print(f"Original buffer: {N} transitions, obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"Stack size: {stack_size}")
    print(f"New obs_dim: {obs_dim * stack_size} = {obs_dim} x {stack_size}")

    # Find episode boundaries
    if episode_index is not None:
        # Use episode_index to find boundaries
        unique_episodes = np.unique(episode_index)
        episode_starts = []
        episode_ends = []
        for ep in unique_episodes:
            ep_mask = episode_index == ep
            ep_indices = np.where(ep_mask)[0]
            episode_starts.append(ep_indices[0])
            episode_ends.append(ep_indices[-1])
        episode_starts = np.array(episode_starts)
        episode_ends = np.array(episode_ends)
    else:
        # Fallback: use dones/terminals
        episode_ends = np.where(dones)[0]
        episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])

    print(f"Found {len(episode_ends)} episodes")

    # Create stacked observations
    stacked_obs = []
    stacked_next_obs = []
    stacked_actions = []
    stacked_rewards = []
    stacked_dones = []

    # Process each episode separately to avoid stacking across episodes
    for ep_idx, (start, end) in enumerate(tqdm(
        zip(episode_starts, episode_ends),
        total=len(episode_ends),
        desc="Processing episodes"
    )):
        ep_len = end - start + 1

        # Skip episodes shorter than stack_size
        if ep_len < stack_size:
            print(f"  Skipping episode {ep_idx} (length {ep_len} < stack_size {stack_size})")
            continue

        # Create frame buffer for this episode
        # Initialize with zeros for padding at the start
        frame_buffer = np.zeros((stack_size, obs_dim), dtype=np.float32)

        for t in range(ep_len):
            idx = start + t

            # Shift buffer left and add new frame at the end
            # Result: frame_buffer = [obs(t-3), obs(t-2), obs(t-1), obs(t)] for stack_size=4
            frame_buffer[:-1] = frame_buffer[1:]
            frame_buffer[-1] = obs[idx]

            # Stack frames: [oldest, ..., newest] -> flatten to (72 * stack_size,)
            # For t=0,1,2: zero-padded (earlier frames are zeros)
            # For t>=3: full history available
            stacked_s = frame_buffer.flatten()

            # Create stacked next_obs (shift and add next_obs)
            next_frame_buffer = frame_buffer.copy()
            next_frame_buffer[:-1] = next_frame_buffer[1:]
            next_frame_buffer[-1] = next_obs[idx]
            stacked_ns = next_frame_buffer.flatten()

            stacked_obs.append(stacked_s)
            stacked_next_obs.append(stacked_ns)
            stacked_actions.append(actions[idx])
            stacked_rewards.append(rewards[idx])
            stacked_dones.append(dones[idx])

    # Convert to numpy arrays
    stacked_obs = np.array(stacked_obs, dtype=np.float32)
    stacked_next_obs = np.array(stacked_next_obs, dtype=np.float32)
    stacked_actions = np.array(stacked_actions, dtype=np.float32)
    stacked_rewards = np.array(stacked_rewards, dtype=np.float32)
    stacked_dones = np.array(stacked_dones, dtype=np.float32)

    print(f"\nStacked buffer statistics:")
    print(f"  Transitions: {len(stacked_obs)}")
    print(f"  Obs shape: {stacked_obs.shape}")
    print(f"  Actions shape: {stacked_actions.shape}")
    print(f"  Obs range: [{stacked_obs.min():.4f}, {stacked_obs.max():.4f}]")
    print(f"  Actions range: [{stacked_actions.min():.4f}, {stacked_actions.max():.4f}]")

    # Save
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        obs=stacked_obs,
        actions=stacked_actions,
        rewards=stacked_rewards,
        next_obs=stacked_next_obs,
        dones=stacked_dones,
        # Metadata
        stack_size=stack_size,
        original_obs_dim=obs_dim,
        stacked_obs_dim=obs_dim * stack_size,
    )
    print("Done!")

    return stacked_obs.shape


def main():
    parser = argparse.ArgumentParser(description="Create frame-stacked replay buffer")
    parser.add_argument("--input", type=str, default="replay_buffer_iql_72d.npz",
                        help="Input replay buffer path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto-generated)")
    parser.add_argument("--stack_size", type=int, default=4,
                        help="Number of frames to stack (default: 4)")

    args = parser.parse_args()

    # Auto-generate output path if not specified
    if args.output is None:
        base = args.input.replace(".npz", "")
        args.output = f"{base}_stacked{args.stack_size}.npz"

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Stack size: {args.stack_size}")
    print()

    create_frame_stacked_buffer(
        input_path=args.input,
        output_path=args.output,
        stack_size=args.stack_size,
    )


if __name__ == "__main__":
    main()
