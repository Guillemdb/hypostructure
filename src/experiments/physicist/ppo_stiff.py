"""
PPO + critic geometry regularization (stiffness + eikonal).

This is the "stiff" ablation: Zeno is disabled.
"""

from hypo_ppo import build_parser, config_from_args, train


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.lambda_zeno = 0.0
    config = config_from_args(args)
    train(config)


if __name__ == "__main__":
    main()
