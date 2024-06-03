from agents.ue5_fbe import UE5_FBE_Env_Agent


def make_ue5_envs_fbe(args):
    envs = UE5_FBE_Env_Agent(args, rank=1)
    return envs

