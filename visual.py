# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:19:26 2021

@author: mm16jdc
"""

import numpy as np


def show_input_target_pair_napari(gen_training, gen_validation=None):
    """
    Press 't' to get a random sample of the next training batch.
    Press 'v' to get a random sample of the next validation batch.
    """
    # Batch
    x, y = next(iter(gen_training))

    # Napari
    import napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        img = viewer.add_image(x, name='input_training')
        tar = viewer.add_labels(y, name='target_training')

        @viewer.bind_key('t')
        def next_batch_training(viewer):
            x, y = next(iter(gen_training))
            img.data = x
            tar.data = y
            img.name = 'input_training'
            tar.name = 'target_training'

        if gen_validation:
            @viewer.bind_key('v')
            def next_batch_validation(viewer):
                x, y = next(iter(gen_validation))
                img.data = x
                tar.data = y
                img.name = 'input_validation'
                tar.name = 'target_validation'

    return viewer


class Input_Target_Pair_Generator:
    def __init__(self,
                 dataloader,
                 re_normalize=True,
                 rgb=False,
                 ):
        self.dataloader = dataloader
        self.re_normalize = re_normalize
        self.rgb = rgb

    def __iter__(self):
        return self

    def __next__(self):
        x, y = next(iter(self.dataloader))
        x, y = x.cpu().numpy(), y.cpu().numpy()  # make sure it's a numpy.ndarray on the cpu

        # Batch
        batch_size = x.shape[0]
        rand_num = np.random.randint(low=0, high=batch_size)
        x, y = x[rand_num], y[rand_num]  # Pick a random image from the batch

        # RGB
        if self.rgb:
            x = np.moveaxis(x, source=0, destination=-1)  # from [C, H, W] to [H, W, C]

        # Re-normalize
        if self.re_normalize:
            from transformations import re_normalize
            x = re_normalize(x)

        return x, y