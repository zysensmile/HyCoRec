# -*- encoding: utf-8 -*-
# @Time    :   2021/5/26
# @Author  :   Chenzhan Shang
# @email   :   czshang@outlook.com

import pickle
import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt


class HyCoRecDataLoader(BaseDataLoader):
    """Dataloader for model KBRD.

    Notes:
        You can set the following parameters in config:

        - ``"context_truncate"``: the maximum length of context.
        - ``"response_truncate"``: the maximum length of response.
        - ``"entity_truncate"``: the maximum length of mentioned entities in context.

        The following values must be specified in ``vocab``:

        - ``"pad"``
        - ``"start"``
        - ``"end"``
        - ``"pad_entity"``

        the above values specify the id of needed special token.

    """

    def __init__(self, opt, dataset, vocab):
        """

        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset)
        self.pad_token_idx = vocab["tok2ind"]["__pad__"]
        self.start_token_idx = vocab["tok2ind"]["__start__"]
        self.end_token_idx = vocab["tok2ind"]["__end__"]
        self.split_token_idx = vocab["tok2ind"].get("_split_", None)
        self.related_truncate = opt.get("related_truncate", None)
        self.context_truncate = opt.get("context_truncate", None)
        self.response_truncate = opt.get("response_truncate", None)
        self.entity_truncate = opt.get("entity_truncate", None)
        self.review_entity2id = vocab["entity2id"]
        return

    def rec_process_fn(self):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict["role"] == "Recommender":
                for item in conv_dict["items"]:
                    augment_conv_dict = {
                        "conv_id": conv_dict["conv_id"],
                        "related_item": conv_dict["item"],
                        "related_entity": conv_dict["entity"],
                        "related_word": conv_dict["word"],
                        "item": item,
                    }
                    augment_dataset.append(augment_conv_dict)

        return augment_dataset

    def rec_batchify(self, batch):
        batch_related_item = []
        batch_related_entity = []
        batch_related_word = []
        batch_movies = []
        batch_conv_id = []
        for conv_dict in batch:
            batch_related_item.append(conv_dict["related_item"])
            batch_related_entity.append(conv_dict["related_entity"])
            batch_related_word.append(conv_dict["related_word"])
            batch_movies.append(conv_dict["item"])
            batch_conv_id.append(conv_dict["conv_id"])

        res = {
            "conv_id": batch_conv_id,
            "related_item": batch_related_item,
            "related_entity": batch_related_entity,
            "related_word": batch_related_word,
            "item": torch.tensor(batch_movies, dtype=torch.long),
        }

        return res

    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch):
        batch_related_tokens = []
        batch_context_tokens = []

        batch_related_item = []
        batch_related_entity = []
        batch_related_word = []

        batch_response = []
        batch_conv_id = []
        for conv_dict in batch:
            batch_related_tokens.append(
                truncate(conv_dict["tokens"][-1], self.related_truncate, truncate_tail=False)
            )
            batch_context_tokens.append(
                truncate(merge_utt(
                    conv_dict["tokens"],
                    start_token_idx=self.start_token_idx,
                    split_token_idx=self.split_token_idx,
                    final_token_idx=self.end_token_idx
                ), self.context_truncate, truncate_tail=False)
            )

            batch_related_item.append(conv_dict["item"])
            batch_related_entity.append(conv_dict["entity"])
            batch_related_word.append(conv_dict["word"])

            batch_response.append(
                add_start_end_token_idx(truncate(conv_dict["response"], self.response_truncate - 2),
                                        start_token_idx=self.start_token_idx,
                                        end_token_idx=self.end_token_idx))
            batch_conv_id.append(conv_dict["conv_id"])

        res = {
            "related_tokens": padded_tensor(batch_related_tokens, self.pad_token_idx, pad_tail=False),
            "context_tokens": padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
            "related_item": batch_related_item,
            "related_entity": batch_related_entity,
            "related_word": batch_related_word,
            "response": padded_tensor(batch_response, self.pad_token_idx),
            "conv_id": batch_conv_id,
        }

        return res

    def policy_batchify(self, *args, **kwargs):
        pass
