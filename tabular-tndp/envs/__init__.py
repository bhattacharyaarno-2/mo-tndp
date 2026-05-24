from gymnasium.envs.registration import register

register(
    id="motndp_amsterdam-v0",
    entry_point="motndp.motndp:MOTNDP",
)

register(
    id="motndp_amsterdam_10x10-v0",
    entry_point="motndp.motndp:MOTNDP",
)

register(
    id="motndp_xian-v0",
    entry_point="motndp.motndp:MOTNDP",
)

register(
    id="motndp_dilemma-v0",
    entry_point="motndp.motndp:MOTNDP",
)

register(
    id="motndp_margins-v0",
    entry_point="motndp.motndp:MOTNDP",
)

