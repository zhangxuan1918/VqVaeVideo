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


def list_videos2(video_folder):
    """
    folder structure
    parent_folder
        - video1
        - video2
    :return:
    """

    files = [os.path.join(video_folder, f2) for f2 in os.listdir(video_folder) if f2.endswith('mp4')]
    return files


params = {
    'image_checkpoint_path': f'/opt/project/data/trained_image/2021-05-25/checkpoint80000.pth.tar',
    'model_args': {
        'group_count': 4,
        'n_hid': 512,
        'vocab_size': 8192,
        'n_init': 768,
        'commitment_cost': 0.25,
        'decay': 0.99,
        'sequence_length': 32
    },
    'data_args': {
        'batch_size': 8,
        'root_dir': '/data/Doraemon/np_arrays/256x256',
        'num_workers': 8,
        'sequence_length': 32,
        'padding_file': 'black_images_code_2021-05-25.npy.gz'
    },
    'train_args': {
        'num_steps': 160000,
        'lr': 1e-4,
        'lr_decay': 0.98,
        'folder_name': '/opt/project/data/trained_video2/' + datetime.datetime.today().strftime('%Y-%m-%d'),
        # 'folder_name': '/opt/project/data/trained_video2/2021-10-16',
        # 'checkpoint_path': '/opt/project/data/trained_video2/2021-06-13/checkpoint66000.pth.tar',
        'checkpoint_path': None
    },
    'use_wandb': False
}


@pipeline_def
def video_pipe(filenames):
    videos = fn.readers.video(device="gpu", filenames=filenames, sequence_length=params['data_args']['sequence_length'],
                              shard_id=params['data_args']['shard_id'],
                              num_shards=params['data_args']['num_shards'], random_shuffle=True,
                              initial_fill=params['data_args']['initial_prefetch_size'])
    return videos
