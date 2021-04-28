import datetime
import os

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn


def list_videos(video_folder):
    """
    folder structure
    parent_folder
        - subfolder1:
            - video1
        - subfolder2:
            - video2
    :return:
    """
    sub_folders = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]
    files = [os.path.join(video_folder, f1, f2) for f1 in sub_folders for f2 in os.listdir(f1) if f2.endswith('mp4')]
    return files


params = {
    'model_args': {
        'num_hiddens': 128,
        'num_residual_hiddens': 32,
        'num_residual_layers': 2,
        'embedding_dim': 256,
        'num_embeddings': 2048,
        'commitment_cost': 0.25,
        'decay': 0.99
    },
    'data_args': {
        'batch_size': 64,
        'num_threads': 6,
        'device_id': 0,
        'training_data_files': list_videos('/data/Doraemon/video_clips/'),
        'seed': 1987,
        'sequence_length': 16,
        'shard_id': 0,
        'num_shards': 1,
        'initial_prefetch_size': 10240
    },
    'train_args': {
        'num_steps': 250000,
        'lr': 2e-4,
        'folder_name': '/opt/project/data/trained_video/' + datetime.datetime.today().strftime('%Y-%m-%d'),
        # 'folder_name': '/opt/project/data/trained_video/2021-04-22',
        'data_std': 0.1,
        # 'checkpoint_path': '/opt/project/data/trained_video/2021-04-22/checkpoint19000.pth.tar'
        'checkpoint_path': None
    }
}


@pipeline_def
def video_pipe(filenames):
    videos = fn.readers.video(device="gpu", filenames=filenames, sequence_length=params['data_args']['sequence_length'],
                              shard_id=params['data_args']['shard_id'],
                              num_shards=params['data_args']['num_shards'], random_shuffle=True,
                              initial_fill=params['data_args']['initial_prefetch_size'])
    return videos
