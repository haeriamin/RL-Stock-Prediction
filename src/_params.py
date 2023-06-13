
def main():
    # Environment parameters
    env_params = dict(
        hmax = 100,
        initial_amount = 10000,
        transaction_cost_pct = 0,
        state_space = state_space,
        stock_dim = stock_dimension,
        tech_indicator_list = tech_indicator_list,
        action_space = stock_dimension,
        reward_scaling = 1e-1,
    )

    return env_params

if __name__ == '__main__':
    main()
