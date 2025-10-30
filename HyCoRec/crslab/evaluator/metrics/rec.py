# @Time   : 2020/11/30
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

# UPDATE:
# @Time   : 2020/12/2
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com
import math

from crslab.evaluator.metrics.base import AverageMetric, SumMetric


class RECMetric(AverageMetric):
    @staticmethod
    def compute(ranks, label, k) -> 'RECMetric':
        return RECMetric(int(label in ranks[:k]))


class NDCGMetric(AverageMetric):
    @staticmethod
    def compute(ranks, label, k) -> 'NDCGMetric':
        if label in ranks[:k]:
            label_rank = ranks.index(label)
            return NDCGMetric(1 / math.log2(label_rank + 2))
        return NDCGMetric(0)


class MRRMetric(AverageMetric):
    @staticmethod
    def compute(ranks, label, k) -> 'MRRMetric':
        if label in ranks[:k]:
            label_rank = ranks.index(label)
            return MRRMetric(1 / (label_rank + 1))
        return MRRMetric(0)


class CovMetric(SumMetric):
    @staticmethod
    def compute(ranks, label, k) -> 'CovMetric':
        return CovMetric(ranks[:k])