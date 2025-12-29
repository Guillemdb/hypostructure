"""
PPO + Zeno actor smoothness constraint.

This is the "zeno" ablation: critic geometry losses are disabled.
"""

from hypo_ppo import build_parser, config_from_args, train


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.lambda_stiffness = 0.0
    args.lambda_eikonal = 0.0
    config = config_from_args(args)
    train(config)


if __name__ == "__main__":
    main()
