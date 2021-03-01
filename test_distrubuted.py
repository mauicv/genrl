"""
Test dask and npybullet multiprocessing
"""

from gym import Env
import time
from src.batch import BatchJob
from distributed import Client, LocalCluster, as_completed


def test_dask_pybullet():
    """Test Dask and Pybullet integration."""

    def play():
        envs = []
        rewards = []
        for i in range(10):
            env = Env('quadruped', var=0, vis=False)
            envs.append(env)
            rewards.append(0)

        for _ in range(300):
            for env_i, env in enumerate(envs):
                _, reward, done, _ = env \
                    .step(env.action_space.sample())
                rewards[env_i] += reward

        return rewards

    cluster = LocalCluster()
    client = Client(cluster)

    futures = []
    for i in range(8):
        futures.append(client.submit(play, key=f'{play.__name__}{i}'))

    completed = as_completed(futures)

    results = []
    for completed_task in completed:
        results += completed_task.result()


batch_job = BatchJob()


def play(genomes):
    envs = []
    rewards = []
    for i in genomes:
        env = Env('quadruped', var=0, vis=False)
        envs.append(env)
        rewards.append(0)

    for i in range(300):
        for i, env in enumerate(envs):
            _, reward, done, _ = env \
                .step(env.action_space.sample())
            rewards[i] += reward
    return rewards


batch_play = batch_job(play)


if __name__ == "__main__":
    start = time.time()
    batch_play([['' for _ in range(10)] for _ in range(8)])
    end = time.time()
    print('multiprocessing time:', end - start)

    start = time.time()
    test_dask_pybullet()
    end = time.time()
    print('dask time:', end - start)
