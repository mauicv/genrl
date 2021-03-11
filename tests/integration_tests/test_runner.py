from simple_neat_env import test_neat_xor
from parallelism_test import test_parallelism
from cart_pole_neat_env import test_cart_pole
from bip_walker_neat_env import test_bip_walker
from load_and_play import play_bip_walker

for test in [
        # test_neat_xor,
        # test_parallelism,
        # test_cart_pole,
        # test_bip_walker,
        play_bip_walker
            ]:
    assert test()
