# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/24
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

# @Time   : 2021/10/06
# @Author : Zhipeng Zhao
# @Email  : oran_official@outlook.com

import torch
from loguru import logger

from .crs import *

Model_register_table = {
    'HyCoRec': HyCoRecModel,
}


def get_model(config, model_name, device, vocab, side_data=None):
    if model_name in Model_register_table:
        model = Model_register_table[model_name](config, device, vocab, side_data)
        logger.info(f'[Build model {model_name}]')
        if config.opt["gpu"] == [-1]:
            return model
        else:
            return torch.nn.DataParallel(model, device_ids=config["gpu"])

    else:
        raise NotImplementedError('Model [{}] has not been implemented'.format(model_name))
